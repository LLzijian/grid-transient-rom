"""Audit one frozen JRun result from saved arrays. No production imports."""
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path("results/gftd_joint_distinguishability_20261112")
EPS = np.finfo(float).eps
K = 1.737409701777866
H = 2.5e-5
ALPHAS = (1 / 64, 1 / 128)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(ref):
    raw = Path(ref["path"]).read_bytes()
    assert sha(raw) == ref["sha256"] and len(raw) == ref["bytes"], ref["path"]
    return raw


def frozen(path, digest):
    raw = path.read_bytes()
    assert sha(raw) == digest, path
    return json.loads(raw)


def norm(value):
    return float(np.linalg.norm(value))


def rms(value):
    return norm(value) / math.sqrt(value.size)


def rounding(value):
    return 64 * EPS * max(1., float(np.max(np.abs(value))))


def same(a, b):
    assert np.allclose(a, b, rtol=2e-13, atol=2e-15), (a, b)


def metric(saved, actual):
    if saved is None:
        assert not math.isfinite(actual)
    else:
        same(saved, actual)


def mapping(saved, x, beta):
    target = x + beta, beta
    for label, wanted in zip(("c", "beta"), target, strict=True):
        actual = np.asarray(saved["actual"][("c", "beta").index(label)])
        assert np.array_equal(saved["target"][("c", "beta").index(label)], wanted)
        assert np.isfinite(actual).all() and np.array_equal(actual > 0, wanted > 0)
        assert np.all(actual[wanted == 0] == 0)
        assert np.all(abs(actual - wanted) <= 8 * EPS * np.maximum(abs(wanted), np.finfo(float).tiny))
    c, b = np.asarray(saved["actual"])
    assert np.all((0 <= b) & (b <= c))


def feasible(x, b, u):
    return all(np.isfinite(v).all() for v in (x, b, u)) and np.all(x >= 0) and np.all(b >= 0) and np.all(abs(u) <= 2)


def shift(x, beta, u, db, direction, delta):
    xx, bb, uu = x.copy(), beta.copy(), u.copy()
    if direction < 45:
        xx[direction] += K * delta
    elif direction < 65:
        uu[direction - 45] += delta
    else:
        bb += delta * db
    return xx, bb, uu


def derivative(r0, a, b, h, sign):
    if not sign:
        return (a - b) / (2 * h), (rounding(a) + rounding(b)) / (2 * h)
    return sign * (-3 * r0 + 4 * a - b) / (2 * h), (3 * rounding(r0) + 4 * rounding(a) + rounding(b)) / (2 * h)


def compare_columns(a, b, saved):
    x, e = a
    y, f = b
    gap = rms(x - y)
    limit = 1e-6 + .05 * max(rms(x), rms(y))
    finite = np.isfinite(x).all() and np.isfinite(y).all() and all(map(math.isfinite, (gap, limit, e + f)))
    passed = finite and gap <= limit and e + f <= limit / 10
    for key, val in (("discrepancy", gap), ("limit", limit), ("rounding_sum", e + f)):
        metric(saved[key], val)
    assert saved["finite"] == finite and saved["passed"] == passed
    return passed


def check_solver(solver, A, b, v, lo, hi):
    assert 0 <= solver["problems"] <= 1 and len(solver["lsmr"]) <= 101
    for i, row in enumerate(solver["lsmr"], 1):
        assert row["number"] == i and row["maxiter"] == 1000
        assert row["alias"] == ("initial" if i == 1 else "trf")
        if row["status"] == "returned":
            assert 0 <= row["itn"] <= 1000
    total = sum(row.get("itn", 0) for row in solver["lsmr"])
    assert total <= 101000
    if "result" in solver:
        result = solver["result"]
        assert 0 <= result["nit"] <= 100
        if v is not None:
            assert result["feasible"] and np.all((lo <= v) & (v <= hi)) and np.isfinite(v).all()
            value = norm(A @ v + b)
            same(result["feasible_residual_norm"], value)
            same(result["cost"], .5 * value**2)
    return total


def check_verification(run, data, reference, A, b, v, x, beta, u, db):
    notes = run["amplitudes"]
    stored = data["verification"]
    assert len(notes) == len(stored) <= 2
    attempted = 0
    for idx, (note, evidence) in enumerate(zip(notes, stored, strict=True)):
        alpha = ALPHAS[idx]
        assert note["alpha"] == evidence["alpha"] == alpha
        groups = {}
        for condition in evidence["conditions"]:
            name = condition["name"]
            assert name not in groups and name in ("control", "compensation")
            c = next(r for r in note["conditions"] if r["name"] == name)
            xx = x if name == "control" else x + alpha * K * v[:45]
            uu = u if name == "control" else u + alpha * v[45:].reshape(24, 20)
            bb = beta + alpha * db
            assert feasible(xx, bb, uu) and np.array_equal(condition["u"], uu)
            mapping(condition["mapping"], xx, bb)
            attempted += len(condition["residuals"])
            assert len(condition["residuals"]) == len(c["packets"]) == 24
            for j, (pair, item) in enumerate(zip(condition["residuals"], c["packets"], strict=True)):
                assert item["local_slot"] == j
                if all(q is not None for q in pair):
                    assert item["finite"]
                    metric(item["mesh_rms"], rms(np.asarray(pair[0]) - pair[1]))
                else:
                    assert not item["finite"] and item["mesh_rms"] is None
            if c["status"] == "finite":
                groups[name] = np.asarray(condition["residuals"]).swapaxes(0, 1)
        if len(groups) != 2:
            assert not note["passed"]
            continue
        control, comp = (groups[name] for name in ("control", "compensation"))
        g = np.stack((reference, control, comp))
        control_delta = (control - reference).reshape(2, -1) / math.sqrt(5280)
        comp_delta = (comp - reference).reshape(2, -1) / math.sqrt(5280)
        B = np.linalg.norm(control_delta, axis=1)
        D = np.linalg.norm(comp_delta, axis=1)
        mesh = np.linalg.norm(g[:, 1] - g[:, 0], axis=2) / math.sqrt(220)
        per_packet = np.linalg.norm(comp - reference, axis=2) / math.sqrt(220)
        em = max(norm(row) / math.sqrt(5280) for row in (g[:, 1] - g[:, 0]).reshape(3, -1))
        er = rounding(g)
        eta = em + er
        bound = np.minimum(1e-6, .1 * B)
        linear_ok = []
        for actual, predicted, saved in zip(
            (control_delta[1], comp_delta[1]), (alpha * b, alpha * (A @ v + b)), note["linear"], strict=True
        ):
            gap = norm(actual - predicted)
            tol = 1e-7 + .1 * max(norm(actual), norm(predicted))
            metric(saved["gap"], gap)
            metric(saved["limit"], tol)
            finite = np.isfinite(predicted).all() and math.isfinite(gap) and math.isfinite(tol)
            linear_ok.append(finite and gap <= tol)
            assert saved["finite"] == finite and saved["passed"] == linear_ok[-1]
        finite = all(np.isfinite(q).all() for q in (g, B, D, bound, mesh, per_packet, eta, 100 * eta, 10 * eta))
        gates = {
            "finite": finite, "mesh": finite and np.all(mesh <= 1e-8), "linear": all(linear_ok),
            "signal": finite and B[1] >= max(1e-5, 100 * eta),
            "remaining_effect": finite and np.all(D <= bound) and bound[1] - D[1] >= 10 * eta,
            "packet_effect": finite and np.all(per_packet <= 1e-5),
        }
        assert note["gates"] == gates and note["passed"] == all(gates.values())
        for key, value in (
            ("B", B), ("D", D), ("limits", bound), ("eta", eta), ("eta_mesh", em), ("eta_round", er),
            ("mesh_rms", mesh), ("packet_compensation_rms", per_packet),
        ):
            for got, want in zip(np.asarray(note[key], dtype=object).ravel(), np.asarray(value).ravel(), strict=True):
                metric(got, float(want))
    return attempted


def main(execution_sha, evidence_sha, host_sha):
    execution_path = ROOT / "science/execution.json"
    run = frozen(execution_path, execution_sha)
    host = frozen(ROOT / "science_host_completion.json", host_sha)
    assert run["evidence"]["sha256"] == evidence_sha
    assert run["status"] == "completed" and not run["cost_unknown"]
    assert run["original_data_reads"] == run["future_reads"] == 0
    release = frozen(ROOT / "science_release.json", run["release_sha256"])
    assert release["released"] and all(run[k] == release[k] for k in (
        "run", "phase", "action_id", "source_sha256", "scipy_source_sha256", "runtime",
        "protocol_sha256", "design_adoption_sha256", "qualification", "prepare",
    ))
    assert all(run[k] == v for k, v in json.loads(read(run["claim"])).items())
    prep = json.loads(read(run["prepare"]))
    qual = json.loads(read(run["qualification"]["receipt"]))
    assert prep["status"] == "completed" and qual["status"] == "passed" and qual["real_science_calls"] == 0
    assert all(run[k] == prep[k] == qual[k] for k in ("runtime", "source_sha256", "scipy_source_sha256"))
    assert len(run["source_sha256"]) == 47
    assert all(sha(Path(p).read_bytes()) == h for p, h in run["source_sha256"].items())
    data = json.loads(gzip.decompress(read(run["evidence"])))
    bundle = json.loads(read(prep["bundle"]))
    assert bundle["slots"] == prep["slots"] and len(bundle["slots"]) == 24
    reference_count = len(data.get("reference", []))
    fd_count = 0
    A, b, v = None, None, None
    if "reference_u" in data:
        c, beta, u, db = (np.asarray(data[k]) for k in ("reference_c", "reference_beta", "reference_u", "d_beta"))
        x = c - beta
        assert np.array_equal(u, [row["u"] for row in bundle["points"]])
        assert np.array_equal(data["saved_residual"], [row["residual"] for row in bundle["points"]])
        f = bundle["physics"]
        assert np.array_equal(c, np.asarray(f["k_edge"]) * f["gates"]["s_edge"])
        assert np.array_equal(beta, c * f["gates"]["s_mod"])
        assert np.array_equal(db, beta - bundle["beta_parent"])
        assert feasible(x, beta, u)
        assert all(feasible(x, beta + a * db, u) for a in ALPHAS) or reference_count == 0
    if reference_count:
        assert reference_count == len(run["reference"]) == 24
        for j, ((values, gap), note) in enumerate(zip(data["reference"], run["reference"], strict=True)):
            assert note["local_slot"] == j and note["mesh_rms"] == gap
            saved = np.asarray(data["saved_residual"][j])
            lim = 1e-12 * max(1., np.max(abs(saved)))
            assert note["limit"] == lim
            if values[0] is not None:
                delta = max(abs(np.asarray(values[0]) - saved))
                same(note["main_saved_max_abs"], delta)
            else:
                delta = None
                assert note["main_saved_max_abs"] is None
            if gap is not None:
                same(gap, rms(np.asarray(values[0]) - values[1]))
            assert note["passed"] == (delta is not None and delta <= lim and gap is not None and gap <= 1e-8)
    passed_columns = 0
    if data["columns"]:
        assert all(row["passed"] for row in run["reference"])
        reference = np.asarray([p[0] for p in data["reference"]]).swapaxes(0, 1)
        A, b = np.zeros((5280, 525)), np.zeros(5280)
        for idx, col in enumerate(data["columns"]):
            j, direction = divmod(idx, 66)
            assert (col["local_slot"], col["direction"]) == (j, direction)
            can = lambda d: feasible(*shift(x, beta, u[j], db, direction, d))
            sign = 0 if can(H) and can(-H) else next(k for k in (1, -1) if can(k * H) and can(k * 2 * H))
            assert col["template"] == sign
            stencils = col["stencils"]
            fd_count += len(stencils)
            expected = [t for h in (H, H / 2) for t in ((h, -h) if sign == 0 else (sign * h, sign * 2 * h))]
            for delta, record in zip(expected[:len(stencils)], stencils, strict=True):
                xx, bb, uu = shift(x, beta, u[j], db, direction, delta)
                assert record["displacement"] == delta and feasible(xx, bb, uu)
                if 45 <= direction < 65:
                    assert record["mapping"] is None
                else:
                    mapping(record["mapping"], xx, bb)
                pair = record["residuals"]
                if all(v is not None for v in pair):
                    metric(record["mesh_rms"], rms(np.asarray(pair[0]) - pair[1]))
                else:
                    assert record["mesh_rms"] is None
            if "gates" not in col:
                assert idx == len(data["columns"]) - 1
                continue
            assert len(stencils) == 4 and all(row["mesh_rms"] <= 1e-8 for row in stencils)
            hh = []
            for i, h in enumerate((H, H / 2)):
                ra, rb = [np.asarray(v["residuals"]) for v in stencils[2*i:2*i+2]]
                hh.append([derivative(reference[g][j], ra[g], rb[g], h, sign) for g in range(2)])
            a = compare_columns(hh[0][1], hh[1][1], col["gates"]["two_h"])
            g = compare_columns(hh[1][0], hh[1][1], col["gates"]["two_grids"])
            if not (a and g):
                assert idx == len(data["columns"]) - 1
                continue
            passed_columns += 1
            rows = slice(220*j, 220*(j+1))
            value = hh[1][1][0]
            if direction == 65:
                b[rows] = value
            else:
                A[rows, direction if direction < 45 else direction + j*20] = value
        assert run["passed_columns"] == passed_columns
        if "matrix" in run:
            assert passed_columns == 1584
            A, b = A / np.sqrt(5280), b / np.sqrt(5280)
            for key, arr in (("A", A), ("b", b)):
                assert np.array_equal(arr, data[key])
                assert run["matrix"][key] == dict(shape=list(arr.shape), sha256=sha(arr.astype("<f8").tobytes()))
    verify_count = 0
    lo = hi = None
    if "matrix" in run:
        lo = np.r_[np.maximum(-1., -x/(ALPHAS[0]*K)), np.maximum(-1., (-2-u.ravel())/ALPHAS[0])]
        hi = np.r_[np.ones(45), np.minimum(1., (2-u.ravel())/ALPHAS[0])]
        assert np.array_equal(lo, data["lower"]) and np.array_equal(hi, data["upper"])
        if "v" in data:
            v = np.asarray(data["v"])
            verify_count = check_verification(run, data, reference, A, b, v, x, beta, u, db)
    itn = check_solver(run["solver"], A, b, v, lo, hi)
    assert run["lsmr_iterations"] == itn and run["lsmr_calls"] == len(run["solver"]["lsmr"])
    pairs = {
        "reference": [row[0] for row in data.get("reference", [])],
        "fd": [row["residuals"] for col in data["columns"] for row in col["stencils"]],
        "verification": [
            pair for alpha in data["verification"] for condition in alpha["conditions"]
            for pair in condition["residuals"]
        ],
    }
    for phase, n, cap in (("reference", reference_count, 24), ("fd", fd_count, 6336), ("verification", verify_count, 96)):
        assert n == len(pairs[phase])
        for g, (row, rhs) in enumerate(zip(run["costs"][phase], (480, 960), strict=True)):
            completed = sum(pair[g] is not None for pair in pairs[phase])
            assert not row["cost_unknown"] and row["maps"] == n <= cap
            assert completed * rhs <= row["rhs"] <= n * rhs
    entries = [r for part in run["costs"].values() for r in part]
    assert sum(r["maps"] for r in entries) == run["maps"] <= 12912
    assert sum(r["rhs"] for r in entries) == run["rhs"] <= 9296640
    count = sum(n["passed"] for n in run["amplitudes"])
    assert run["witness"] == ("both" if count == 2 else "one" if count else "unresolved")
    print(json.dumps(dict(execution_sha256=execution_sha, evidence_sha256=evidence_sha, host_sha256=host_sha,
                          passed_columns=passed_columns, maps=run["maps"], rhs=run["rhs"], witness=run["witness"],
                          solver=run["solver"], amplitudes=run["amplitudes"], host_status=host["status"]),
                     ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main(*sys.argv[1:])
