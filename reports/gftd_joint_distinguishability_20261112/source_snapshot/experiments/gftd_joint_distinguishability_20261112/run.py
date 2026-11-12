"""Released cache-only, two-amplitude compensation witness."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import gzip
import importlib
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

if __name__ == "__main__":
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np

from experiments.gftd_objective_alignment_20261112 import run as prior

io, core, fit, engine = prior.evidence, prior.core, prior.fit297, prior.fitting
require, bound_json = prior.require, prior.bound_json
RUN = "gftd_joint_distinguishability_20261112"
RESULTS, PRIVATE = Path("results") / RUN, Path("artifacts") / RUN
PROTOCOL = "77da56d7717036fa20e27fe4012f5795d55d3ffecb40818dbcba71f582b56e10"
ADOPTION = "547d627655192fd56a86b5ff25faf4590dc930e57ff06217d75960a1781a74c2"
PREPARE = "995399315f5d32d8e7f50d3894f7879f1a810e98cae73695581923f5043b7cef"
QUARTER = "7b98ad2b68220d6599baaaa71dced108d574af643bece9ae8bc45ab2e9c5b35a"
CACHE_SHAS = {
    "packets": "1cc6cb649cc9aaa073a5eb996cf279097fe7a968e678f09a4a057edb20aba403",
    "parent": "8c478b04b9dfb005bbfc24d7649aa8108f32482c5e1f92faea91bd8be8a93f51",
    "coefficients": "19e16fd0d7eeb457e10ffe46bde384dcce34bc1354d93d0d9e91016ef527d992",
    "details": "07faaa42dd44e0b0dfa245f18ac079cf1de5649f6f176254d48bd56892c0b0aa",
}
SCIPY_SHAS = {
    "scipy/optimize/_lsq/lsq_linear.py": "9a6e3a1e3f965510ec49c193d86543f897e6d7a89f2962aec4eb9e0b73af3b9d",
    "scipy/optimize/_lsq/trf_linear.py": "184306c8ca96c1a54b51aeec0a8f62c0399492e4fbe3aa1c8b588ae8a8e72500",
    "scipy/sparse/linalg/_isolve/lsmr.py": "73d4b3c402e20dd08e8e7690184253c4b755dc75c2b98f76361dee798596bb9a",
}
ALPHAS, STEPS, H = (1/64, 1/128), (.0025, .00125), 2.5e-5
EPS, TINY = np.finfo(float).eps, np.finfo(float).tiny
LIMITS = {"reference": (24, 24), "fd": (6336, 6336), "verification": (96, 96)}
K_REF = 1.737409701777866


class Unresolved(ValueError):
    """A bounded scientific gate failed with known work."""


def scipy_sources():
    from importlib.metadata import distribution
    dist = distribution("scipy")
    require(dist.version == "1.18.0", "SciPy version changed")
    modules = []
    for path, sha in SCIPY_SHAS.items():
        file = Path(dist.locate_file(path)).resolve()
        require(io.sha(file.read_bytes()) == sha, "SciPy source changed")
        module = importlib.import_module(path[:-3].replace("/", "."))
        require(Path(module.__file__).resolve() == file, "SciPy import identity changed")
        modules.append(module)
    require(modules[0].lsmr is modules[1].lsmr is modules[2].lsmr, "LSMR aliases differ")
    return modules[:2]


def frozen_context(phase, release):
    adoption = json.loads(io.read_bound(RESULTS / "design_adoption.json", ADOPTION))
    refs = adoption["references"]
    for ref in refs.values():
        io.read_ref(ref, Path(ref["path"]))
    require(refs["protocol"]["sha256"] == PROTOCOL, "protocol binding changed")
    archive = bound_json(refs["parent_source_archive"], prior.RESULTS / "source_archive.json")
    sources = {row["source_path"]: row["sha256"] for row in archive["files"]}
    require(len(sources) == 45, "parent source count differs")
    for name in ("run", "qualify"):
        path = f"experiments/{RUN}/{name}.py"
        sources[path] = io.sha(io.local_path(path).read_bytes())
    io.verify_sources({"source_sha256": sources})
    runtime = io.runtime()
    modules = scipy_sources()
    require(release.get("run") == RUN and release.get("phase") == phase
            and release.get("released") is True and isinstance(release.get("action_id"), str)
            and bool(release["action_id"]) and release.get("protocol_sha256") == PROTOCOL
            and release.get("design_adoption_sha256") == ADOPTION and release.get("source_sha256") == sources
            and release.get("runtime") == runtime and release.get("scipy_source_sha256") == SCIPY_SHAS,
            "explicit frozen release identity required")
    return adoption, modules


def cache_ref(ref, expected_path, key):
    require(ref["sha256"] == CACHE_SHAS[key] and ref["path"] == expected_path.as_posix(),
            f"parent {key} identity differs")
    return ref


def parent_bindings(adoption, runtime):
    closure = bound_json(adoption["references"]["parent_closure"], prior.RESULTS / "final_adoption.json")
    refs = closure["references"]
    io.read_ref(refs["result_review"], Path(refs["result_review"]["path"]))
    require(refs["result_audit"]["sha256"] ==
            "93a455515512885196cc993c5fcb2758f255768df46b99f11d4178522b291ebb", "parent audit identity differs")
    io.read_ref(refs["result_audit"], Path(refs["result_audit"]["path"]))
    execution = bound_json(refs["execution"], prior.RESULTS / "science/execution.json")
    prepare_ref, job_ref = execution["prepare"], refs["modulated_quarter"]
    require(execution["status"] == "completed" and execution["paired_complete"]
            and execution["runtime"] == runtime and prepare_ref["sha256"] == PREPARE
            and job_ref["sha256"] == QUARTER and job_ref in execution["modes"]["modulated"]["jobs"]
            and execution["modes"]["modulated"]["selection"]["point"] == "quarter", "parent outcome differs")
    prepared = bound_json(prepare_ref, prior.RESULTS / "prepare.json")
    job = bound_json(job_ref, prior.RESULTS / "science/jobs/modulated_quarter.json")
    require(prepared["status"] == "completed" and prepared["runtime"] == runtime
            and prepared["matched_preparation_keys"] == prior.KEYS
            and prepared["old_individual_packet_hashes"] is None
            and job["status"] == "completed" and job["t"] == .25
            and job["qualification_passed"] == 24 and not job["cost_unknown"]
            and job["prepare"] == prepare_ref, "parent prepare or candidate differs")
    slots, hashes = prepared["slots"], prepared["forward_packet_hashes"]
    prior.verify_slots(slots, hashes)
    require(all(len(row["packet_sha256"]) == 64 and
                all(char in "0123456789abcdef" for char in row["packet_sha256"]) for row in hashes),
            "parent forward packet hash differs")
    base = prior.PRIVATE / "science/modulated_quarter"
    caches = {
        "packets": cache_ref(prepared["packets"], prior.PRIVATE / "prepare/packets.json", "packets"),
        "parent": cache_ref(prepared["caches"]["modulated"], prior.PRIVATE / "prepare/modulated.json", "parent"),
        "coefficients": cache_ref(job["coefficients"], base / "coefficients.json", "coefficients"),
        "details": cache_ref(job["details"], base / "details.json.gz", "details"),
    }
    return slots, hashes, caches, prepare_ref, job_ref, job


def prepare(identity, adoption):
    claim = prior.claim(RESULTS / "prepare_claim.json", identity)
    receipt = dict(**identity, claim=claim, status="interrupted", private_cache_opened=False,
                   original_data_reads=0, future_reads=0)
    try:
        slots, hashes, caches, prepare_ref, job_ref, job = parent_bindings(adoption, identity["runtime"])
        receipt["private_cache_opened"] = True
        packet_rows = bound_json(caches["packets"], Path(caches["packets"]["path"]))
        parent = bound_json(caches["parent"], Path(caches["parent"]["path"]))
        coeff = bound_json(caches["coefficients"], Path(caches["coefficients"]["path"]))
        detail = json.loads(gzip.decompress(io.read_ref(caches["details"], Path(caches["details"]["path"]))))
        prior.verify_slots(slots, [row["slot"] for row in packet_rows])
        prior.verify_slots(slots, parent["slots"])
        require(parent["preparation_key"] == prior.KEYS["modulated"] and parent["mode"] == "modulated"
                and coeff["endpoints"][0] == parent["endpoints"][0]
                and coeff["delta_star"] == parent["delta_star"], "parent coefficient cache differs")
        packets = [core.validate_observation(row["packet"], 10) for row in packet_rows]
        require(all(packet["sigma"] == 0 for packet in packets), "noise setting differs")
        config = coeff["inner_config"]
        functional = prior.restore_physics(coeff["physics"])
        fp = engine._hash(engine._physics(functional))
        prior.same_config(config, parent["config"], fp)
        fit.validate_config(functional, config)
        require(config["fit_request"]["k_ref"] == K_REF and fp == job["functional_sha256"]
                and engine._hash(config) == job["config_sha256"], "quarter configuration differs")
        c, beta = prior.effective(functional, "modulated")
        require(np.array_equal(c, coeff["actual"]["c"]) and np.array_equal(beta, coeff["actual"]["beta"]),
                "quarter effective coefficients differ")
        candidate = detail["candidate"]
        require(candidate["status"] == "completed" and candidate["model_sha256"] == fp
                and candidate["objective"] == job["objective"], "quarter detail identity differs")
        events = candidate["events"]
        prior.verify_slots(slots, events)
        points = []
        for event in events:
            best = event["best"]
            require(best == event["tasks"][event["selected"]]["best"]
                    and engine._hash(best["state"]) == event["endpoint_sha256"]
                    and event["qualification_status"] == "passed"
                    and np.shape(best["u"]) == (20,) and np.shape(best["residual"]) == (220,)
                    and np.all(np.isfinite(best["residual"]))
                    and np.all(np.abs(np.asarray(best["u"])) <= 2), "selected quarter point differs")
            points.append(dict(u=best["u"], residual=best["residual"]))
        bundle = dict(physics=coeff["physics"], config=config, points=points, slots=slots,
                      beta_parent=parent["endpoints"][0]["beta"], packet_ref=caches["packets"])
        receipt.update(status="completed", slots=slots, forward_packet_hashes=hashes,
                       inherited_preparation_keys=prior.KEYS, parent_prepare=prepare_ref, quarter_job=job_ref,
                       caches=caches, bundle=io.save_json(PRIVATE / "prepare/bundle.json", bundle, exclusive=True))
    except Exception as error:
        receipt.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        io.save_json(RESULTS / "prepare.json", receipt, exclusive=True)


def bounds(x, u, scale):
    lo = np.r_[np.maximum(-1., -x/(ALPHAS[0]*scale)),
               np.maximum(-1., (-2-u.ravel())/ALPHAS[0])]
    hi = np.r_[np.ones(x.size), np.minimum(1., (2-u.ravel())/ALPHAS[0])]
    require(np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
            and np.all((lo <= 0) & (0 <= hi) & (lo < hi)), "empty or invalid compensation interval")
    return lo, hi


def feasible(x, beta, u):
    return bool(all(np.all(np.isfinite(v)) for v in (x, beta, u))
                and np.all(x >= 0) and np.all(beta >= 0) and np.all(np.abs(u) <= 2))


def mapped(parent, config, x, beta):
    if not feasible(x, beta, np.empty(0)):
        raise Unresolved("infeasible coefficient target")
    target = x+beta, beta
    candidate = engine._parameters(parent, *target)
    actual = prior.effective(candidate, "modulated")
    prior.mapped_support(target, actual, tuple(v > 0 for v in target))
    non_edge = lambda f: engine._hash(engine._physics(f, include_edges=False))
    require(non_edge(candidate) == non_edge(parent), "non-edge coefficients differ")
    physics = engine._plain(engine._physics(candidate))
    fingerprint = engine._hash(physics)
    new = fit.make_config(candidate, "modulated", config["fit_request"], 297, config["preparation_key"])
    prior.same_config(new, config, fingerprint)
    restored = json.loads(io.json_bytes(dict(physics=physics, config=new)))
    candidate = prior.restore_physics(restored["physics"])
    require(engine._hash(engine._physics(candidate)) == fingerprint and restored["config"] == new
            and all(np.array_equal(a, b) for a, b in zip(actual, prior.effective(candidate, "modulated"), strict=True)),
            "mapped functional round trip differs")
    return candidate, new, dict(target=target, actual=actual, functional_sha256=fingerprint)


def rms(v):
    return float(np.linalg.norm(v) / np.sqrt(v.size))


def rounding(v):
    return 64*EPS*max(1., float(np.max(np.abs(v))))


def history(functional, config, packet, u, step):
    local = engine._counts(engine.dop853_step)
    result = None
    try:
        state = fit.endpoint(packet, functional, u, "modulated", "proxy", config)
        dense = engine._integrate(functional, state, step, .1, local, backward=True,
                                  dense=True, stepper=engine.dop853_step)
        raw = fit.history_residual(functional, dense[::(4 if step == STEPS[0] else 8)], packet, config)
        if raw.dtype == np.float64 and raw.shape == (220,) and np.all(np.isfinite(raw)):
            result = raw
    except (FloatingPointError, OverflowError, ValueError):
        pass
    return result, local["actual_rhs_calls"]


class Maps:
    def __init__(self, costs, callback=history):
        self.costs = costs
        self.callback = callback

    def pair(self, functional, config, packet, u, phase):
        require(np.shape(u) == (20,) and np.all(np.isfinite(u)) and np.all(np.abs(u) <= 2), "illegal endpoint")
        values = []
        for index, step in enumerate(STEPS):
            ledger = self.costs[phase][index]
            cap = (480, 960)[index]
            require(ledger["maps"] < LIMITS[phase][index]
                    and ledger["rhs"] + cap <= LIMITS[phase][index]*cap, "history budget exhausted")
            ledger["maps"] += 1
            ledger["cost_unknown"] = True
            raw, rhs = self.callback(functional, config, packet, u, step)
            require(type(rhs) is int and 0 <= rhs <= cap and (raw is None or rhs == cap),
                    "history callback ledger differs")
            ledger["rhs"] += rhs
            ledger["cost_unknown"] = False
            values.append(raw)
        valid = all(v is not None for v in values)
        gap = rms(values[0]-values[1]) if valid else None
        return values, gap


def template(can):
    if can(H) and can(-H):
        return 0
    for sign in (1, -1):
        if can(sign*H) and can(sign*2*H):
            return sign
    raise Unresolved("no feasible finite-difference template")


def difference(ref, first, second, step, sign):
    if sign == 0:
        return (first-second)/(2*step), (rounding(first)+rounding(second))/(2*step)
    return sign*(-3*ref+4*first-second)/(2*step), (
        3*rounding(ref)+4*rounding(first)+rounding(second))/(2*step)


def column_gate(first, second):
    a, ea = first
    b, eb = second
    with np.errstate(over="ignore", invalid="ignore"):
        norm_a, norm_b = rms(a), rms(b)
        discrepancy = rms(a-b)
        gate = 1e-6+.05*max(norm_a, norm_b)
        rounding_sum = ea+eb
    finite = bool(np.all(np.isfinite(a)) and np.all(np.isfinite(b))
                  and np.all(np.isfinite([norm_a, norm_b, discrepancy, gate, rounding_sum])))
    return dict(discrepancy=discrepancy, limit=gate, rounding_sum=rounding_sum, finite=finite,
                passed=finite and discrepancy <= gate and rounding_sum <= gate/10)


def assemble_column(A, b, packet, direction, value):
    rows = slice(220*packet, 220*(packet+1))
    if direction == 65:
        b[rows] = value
    else:
        col = direction if direction < 45 else direction + packet*20
        A[rows, col] = value


def shifted(x, beta, u, d_beta, direction, amount):
    xx, bb, uu = x.copy(), beta.copy(), u.copy()
    if direction < 45:
        xx[direction] += amount*K_REF
    elif direction < 65:
        uu[direction-45] += amount
    else:
        bb += amount*d_beta
    return xx, bb, uu


def linearize(parent, config, packets, x, beta, u, d_beta, reference, maps, private, public):
    A, b = np.zeros((5280, 525)), np.zeros(5280)
    for j, packet in enumerate(packets):
        for direction in range(66):
            def perturb(amount):
                return shifted(x, beta, u[j], d_beta, direction, amount)

            mapped_values = {}

            def construct(amount, xx, bb):
                if 45 <= direction < 65:
                    return parent, config, None
                if amount not in mapped_values:
                    mapped_values[amount] = mapped(parent, config, xx, bb)
                return mapped_values[amount]

            def can(amount):
                xx, bb, uu = perturb(amount)
                if not feasible(xx, bb, uu):
                    return False
                construct(amount, xx, bb)
                return True

            sign = template(can)
            stored = dict(local_slot=j, direction=direction, template=sign, stencils=[])
            private["columns"].append(stored)
            by_h = []
            for h in (H, H/2):
                evaluated = []
                for t in ((h, -h) if sign == 0 else (sign*h, sign*2*h)):
                    xx, bb, uu = perturb(t)
                    if not feasible(xx, bb, uu):
                        raise Unresolved("fixed template became infeasible")
                    functional, inner, mapping = construct(t, xx, bb)
                    values, gap = maps.pair(functional, inner, packet, uu, "fd")
                    stored["stencils"].append(dict(displacement=t, mapping=mapping, residuals=values, mesh_rms=gap))
                    if gap is None or gap > 1e-8:
                        raise Unresolved("finite-difference history mesh gate failed")
                    evaluated.append(values)
                with np.errstate(over="ignore", invalid="ignore"):
                    derivatives = [difference(reference[g][j], evaluated[0][g], evaluated[1][g], h, sign)
                                   for g in range(2)]
                if any(not np.all(np.isfinite(col)) or not np.isfinite(indicator)
                       for col, indicator in derivatives):
                    public["failed_column"] = dict(local_slot=j, direction=direction, template=sign,
                                                  h=h, reason="nonfinite_difference")
                    raise Unresolved("nonfinite finite-difference column")
                by_h.append(derivatives)
            fine_step = by_h[1][1]
            gates = dict(two_h=column_gate(by_h[0][1], fine_step), two_grids=column_gate(by_h[1][0], fine_step))
            stored["gates"] = gates
            if not all(v["passed"] for v in gates.values()):
                public["failed_column"] = dict(local_slot=j, direction=direction, template=sign, **gates)
                raise Unresolved("finite-difference column gate failed")
            public["passed_columns"] += 1
            assemble_column(A, b, j, direction, fine_step[0])
    A, b = A/np.sqrt(5280), b/np.sqrt(5280)
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
        raise Unresolved("nonfinite assembled linearization")
    return A, b


def feasible_vector(v, lower, upper):
    return bool(np.shape(v) == lower.shape and np.all(np.isfinite(v))
                and np.all((lower <= v) & (v <= upper)))


def solve(A, b, lower, upper, modules, ledger, *, outer=100, inner=1000):
    require(ledger["problems"] == 0 and (outer, inner) in ((100, 1000), (10, 20)), "solver budget exhausted")
    require(A.shape == (b.size, lower.size) and lower.shape == upper.shape, "linear problem shape differs")
    if not all(np.all(np.isfinite(value)) for value in (A, b, lower, upper)):
        raise Unresolved("nonfinite linear problem")
    ledger["problems"] += 1
    first, trf = modules

    def guarded(alias, original):
        def call(*args, **kwargs):
            require(kwargs.get("maxiter") == inner and len(ledger["lsmr"]) < outer+1
                    and sum(row.get("itn", 0) for row in ledger["lsmr"]) + inner <= (outer+1)*inner,
                    "LSMR budget exhausted")
            row = dict(alias=alias, number=len(ledger["lsmr"])+1, maxiter=inner, status="interrupted")
            ledger["lsmr"].append(row)
            result = original(*args, **kwargs)
            _, istop, itn, normr, normar, normA, condA, normx = result
            require(0 <= itn <= inner, "LSMR returned invalid iteration count")
            row.update(status="returned", istop=int(istop), itn=int(itn), normr=float(normr), normar=float(normar),
                       normA=float(normA), condA=float(condA), normx=float(normx))
            return result
        return call

    with ExitStack() as stack:
        for alias, module in (("initial", first), ("trf", trf)):
            stack.enter_context(patch.object(module, "lsmr", guarded(alias, module.lsmr)))
        result = first.lsq_linear(A, -b, bounds=(lower, upper), method="trf", lsq_solver="lsmr",
                                  tol=1e-10, lsmr_tol=1e-12, max_iter=outer, lsmr_maxiter=inner)
    require(0 <= result.nit <= outer, "TRF returned invalid iteration count")
    valid = feasible_vector(result.x, lower, upper)
    ledger["result"] = dict(status=int(result.status), nit=int(result.nit), message=str(result.message),
                            cost=float(result.cost), optimality=float(result.optimality), feasible=valid,
                            feasible_residual_norm=float(np.linalg.norm(A@result.x+b)) if valid else None)
    return result.x if valid else None


def verification_metrics(reference, control, compensation, alpha, linear_control, linear_comp):
    groups = np.stack((reference, control, compensation))
    scale = np.sqrt(5280)
    with np.errstate(over="ignore", invalid="ignore"):
        b = (control-reference).reshape(2, -1)/scale
        d = (compensation-reference).reshape(2, -1)/scale
        B, D = np.linalg.norm(b, axis=1), np.linalg.norm(d, axis=1)
        eta_mesh = float(np.max(np.linalg.norm((groups[:, 1]-groups[:, 0]).reshape(3, -1)/scale, axis=1)))
        eta_round = rounding(groups)
        eta = eta_mesh + eta_round
        mesh = np.linalg.norm(groups[:, 1]-groups[:, 0], axis=2)/np.sqrt(220)
        packet_diff = np.linalg.norm(compensation-reference, axis=2)/np.sqrt(220)
        limits = np.minimum(1e-6, .1*B)
        signal_limit, margin = 100*eta, 10*eta
        linear = []
        for actual, predicted in ((b[1], alpha*linear_control), (d[1], alpha*linear_comp)):
            gap = float(np.linalg.norm(actual-predicted))
            threshold = float(1e-7+.1*max(np.linalg.norm(actual), np.linalg.norm(predicted)))
            finite = bool(np.all(np.isfinite(predicted)) and np.all(np.isfinite([gap, threshold])))
            linear.append(dict(gap=gap, limit=threshold, finite=finite, passed=finite and gap <= threshold))
    finite = all(np.all(np.isfinite(value)) for value in
                 (groups, B, D, limits, mesh, packet_diff, eta, signal_limit, margin))
    gates = dict(finite=finite, mesh=finite and bool(np.all(mesh <= 1e-8)),
                 linear=all(row["passed"] for row in linear),
                 signal=finite and bool(B[1] >= 1e-5 and B[1] >= signal_limit),
                 remaining_effect=finite and bool(np.all(D <= limits) and limits[1]-D[1] >= margin),
                 packet_effect=finite and bool(np.all(packet_diff <= 1e-5)))
    return dict(passed=all(gates.values()), gates=gates, B=B.tolist(), D=D.tolist(), limits=limits.tolist(),
                eta=eta, eta_mesh=eta_mesh, eta_round=eta_round, mesh_rms=mesh.tolist(),
                packet_compensation_rms=packet_diff.tolist(), linear=linear)


def verify_amplitudes(parent, config, packets, x, beta, u, d_beta, reference, maps, A, b, v, private, public):
    with np.errstate(over="ignore", invalid="ignore"):
        linear_comp = A@v+b
    for alpha in ALPHAS:
        groups = []
        note = dict(alpha=alpha, conditions=[])
        public["amplitudes"].append(note)
        stored = dict(alpha=alpha, conditions=[])
        private["verification"].append(stored)
        for name in ("control", "compensation"):
            xx = x if name == "control" else x+alpha*K_REF*v[:45]
            uu = u if name == "control" else u+alpha*v[45:].reshape(24, 20)
            bb = beta+alpha*d_beta
            condition = dict(name=name, packets=[])
            note["conditions"].append(condition)
            if not feasible(xx, bb, uu):
                condition["status"] = "infeasible"
                groups.append(None)
                continue
            functional, inner, mapping = mapped(parent, config, xx, bb)
            evidence = dict(name=name, mapping=mapping, u=uu, residuals=[])
            stored["conditions"].append(evidence)
            results = []
            for j, packet in enumerate(packets):
                values, gap = maps.pair(functional, inner, packet, uu[j], "verification")
                results.append(values)
                evidence["residuals"].append(values)
                condition["packets"].append(dict(local_slot=j, mesh_rms=gap, finite=gap is not None))
            valid = all(row["finite"] for row in condition["packets"])
            condition["status"] = "finite" if valid else "nonfinite"
            groups.append(np.asarray(results).swapaxes(0, 1) if valid else None)
        if all(group is not None for group in groups):
            note.update(verification_metrics(reference, *groups, alpha, b, linear_comp))
        else:
            note["passed"] = False


def check_reference(saved, references):
    checks = []
    for j, (values, gap) in enumerate(references):
        limit = 1e-12*max(1., float(np.max(np.abs(saved[j]))))
        error = None if values[0] is None else float(np.max(np.abs(values[0]-saved[j])))
        checks.append(dict(local_slot=j, main_saved_max_abs=error, limit=limit, mesh_rms=gap,
                           passed=error is not None and error <= limit and gap is not None and gap <= 1e-8))
    return checks


def write_private(private):
    return io.save(PRIVATE / "science/evidence.json.gz",
                   gzip.compress(io.json_bytes(prior.matched._json_value(private)), mtime=0), exclusive=True)


def science(identity, prepared, modules):
    stage_claim = prior.claim(RESULTS / "science/claim.json", identity)
    costs = {phase: [dict(maps=0, rhs=0, cost_unknown=False) for _ in STEPS] for phase in LIMITS}
    solver = dict(problems=0, lsmr=[])
    public = dict(**identity, claim=stage_claim, status="interrupted", costs=costs, solver=solver,
                  original_data_reads=0, future_reads=0, passed_columns=0, amplitudes=[])
    private = dict(columns=[], verification=[])
    wall, cpu = time.perf_counter(), time.process_time()
    try:
        data = bound_json(prepared["bundle"], PRIVATE / "prepare/bundle.json")
        require(data["slots"] == prepared["slots"] and data["packet_ref"] == prepared["caches"]["packets"],
                "prepared packet identity differs")
        packet_rows = bound_json(data["packet_ref"], prior.PRIVATE / "prepare/packets.json")
        prior.verify_slots(prepared["slots"], [row["slot"] for row in packet_rows])
        packets = [core.validate_observation(row["packet"], 10) for row in packet_rows]
        require(all(packet["sigma"] == 0 for packet in packets), "prepared noise identity differs")
        parent, config = prior.restore_physics(data["physics"]), data["config"]
        fit.validate_config(parent, config)
        require(config["fit_request"]["k_ref"] == K_REF, "frozen k_ref differs")
        c, beta = prior.effective(parent, "modulated")
        x, d_beta = c-beta, beta-np.asarray(data["beta_parent"], dtype=float)
        u = np.asarray([point["u"] for point in data["points"]], dtype=float)
        saved = np.asarray([point["residual"] for point in data["points"]], dtype=float)
        require(u.shape == (24, 20) and saved.shape == (24, 220) and np.all(np.isfinite(saved))
                and feasible(x, beta, u), "saved point identity differs")
        private.update(reference_c=c, reference_beta=beta, d_beta=d_beta, reference_u=u, saved_residual=saved)
        if not np.any(d_beta):
            raise Unresolved("empty direction")
        if not all(feasible(x, beta+alpha*d_beta, u) for alpha in ALPHAS):
            raise Unresolved("infeasible beta direction")
        lower, upper = bounds(x, u, K_REF)
        maps = Maps(costs)
        references = private["reference"] = []
        for packet, point in zip(packets, u, strict=True):
            references.append(maps.pair(parent, config, packet, point, "reference"))
        public["reference"] = check_reference(saved, references)
        if not all(check["passed"] for check in public["reference"]):
            raise Unresolved("reference cache or mesh gate failed")
        reference = np.asarray([row[0] for row in references]).swapaxes(0, 1)
        A, b = linearize(parent, config, packets, x, beta, u, d_beta, reference, maps, private, public)
        private.update(A=A, b=b, lower=lower, upper=upper)
        public["matrix"] = {name: dict(shape=list(value.shape), sha256=io.sha(value.astype("<f8").tobytes()))
                            for name, value in (("A", A), ("b", b))}
        try:
            v = solve(A, b, lower, upper, modules, solver)
        except (FloatingPointError, np.linalg.LinAlgError) as error:
            raise Unresolved("linear solver failed") from error
        if v is None:
            raise Unresolved("solver returned no feasible finite vector")
        private["v"] = v
        verify_amplitudes(parent, config, packets, x, beta, u, d_beta, reference, maps, A, b, v, private, public)
        public.update(status="completed", witness="both" if all(row["passed"] for row in public["amplitudes"])
                      else "one" if any(row["passed"] for row in public["amplitudes"]) else "unresolved")
    except Unresolved as error:
        public.update(status="completed", witness="unresolved", reason=str(error))
    except Exception as error:
        public.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        entries = [entry for rows in costs.values() for entry in rows]
        public.update(maps=sum(row["maps"] for row in entries), rhs=sum(row["rhs"] for row in entries),
                      cost_unknown=any(row["cost_unknown"] for row in entries)
                      or any(row["status"] != "returned" for row in solver["lsmr"]),
                      lsmr_calls=len(solver["lsmr"]), lsmr_iterations=sum(row.get("itn", 0) for row in solver["lsmr"]),
                      host_wall_seconds=time.perf_counter()-wall, host_cpu_seconds=time.process_time()-cpu)
        try:
            public["evidence"] = write_private(private)
        except Exception as error:
            public.update(status="interrupted", evidence_error_type=type(error).__name__)
            raise
        finally:
            io.save_json(RESULTS / "science/execution.json", prior.matched._json_value(public), exclusive=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "science"))
    parser.add_argument("--release", required=True, type=Path)
    args = parser.parse_args()
    raw = io.local_path(args.release).read_bytes()
    release = json.loads(raw)
    adoption, modules = frozen_context(args.phase, release)
    identity = {key: release[key] for key in ("run", "phase", "action_id", "protocol_sha256", "design_adoption_sha256",
                                             "source_sha256", "runtime", "scipy_source_sha256", "qualification")}
    identity["release_sha256"] = io.sha(raw)
    qualification = release["qualification"]
    attempt = qualification["attempt"]
    require(type(attempt) is int and 1 <= attempt <= 3, "invalid qualification attempt")
    receipt = bound_json(qualification["receipt"], RESULTS / f"qualification_{attempt}.json")
    require(receipt["status"] == "passed" and receipt["attempt"] == attempt and receipt["run"] == RUN
            and receipt["protocol_sha256"] == PROTOCOL and receipt["source_sha256"] == release["source_sha256"]
            and receipt["scipy_source_sha256"] == SCIPY_SHAS and receipt["runtime"] == release["runtime"]
            and receipt["real_science_calls"] == 0 and not receipt["forbidden_attempts"] and len(receipt["groups"]) == 6
            and not receipt["cost_unknown"] and receipt["linear_problems"] == 1
            and 1 <= receipt["lsmr_calls"] <= 11 and 0 <= receipt["lsmr_iterations"] <= 220
            and 0 <= receipt["mock_callbacks"] <= 1000 and 0 <= receipt["cumulative_mock_callbacks"] <= 3000,
            "qualification binding differs")
    if args.phase == "prepare":
        prepare(identity, adoption)
        return
    ref = release["prepare"]
    prepared = bound_json(ref, RESULTS / "prepare.json")
    require(prepared["status"] == "completed" and prepared["phase"] == "prepare"
            and all(prepared[key] == identity[key] for key in ("run", "protocol_sha256", "design_adoption_sha256",
                    "source_sha256", "runtime", "scipy_source_sha256", "qualification"))
            and prepared["inherited_preparation_keys"] == prior.KEYS and prepared["original_data_reads"] == 0
            and prepared["future_reads"] == 0, "preparation identity differs")
    science(dict(**identity, prepare=ref), prepared, modules)


if __name__ == "__main__":
    main()
