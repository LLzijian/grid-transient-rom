"""Frozen-equation, anchored one-step history inversion; no parameter updates."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter, process_time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import numpy as np
import torch
from kiki.models.osgkd import OSGKD
from kiki.models.osgkd_fitting import IntegrationWork, integrate
from kiki.evaluation.osgkd import score_event, summarize, numerical_gate

spec = importlib.util.spec_from_file_location("p2a_parent", ROOT / "experiments/osgkd_v2_p1a_20261123/run.py")
p1a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p1a)
old = p1a.old
RUN = "osgkd_v2_p2a_20261123"
BASE = "artifacts/osgkd_v2_p1a_20261123/pair01/8s"
PAIR = "results/osgkd_v2_p1a_20261123/pair01/summary.json"
AUDIT = "results/osgkd_fit297_20261113/port_audit_release.json"
SOURCES = {f"experiments/{RUN}/run.py", "experiments/osgkd_v2_p1a_20261123/run.py",
           "experiments/osgkd_fit297_20261113/run.py", "campaigns/osgkd_v2/p2a_protocol.md",
           "src/kiki/models/osgkd.py", "src/kiki/models/osgkd_fitting.py",
           "src/kiki/evaluation/osgkd.py", "src/kiki/data/osgkd.py", "experiments/evaluate_l1_latent.py"}


def history_path(model, state, step, work):
    """Chronological backward reconstruction, retaining the observed endpoint."""
    values = [state]
    substeps = round(.01 / step)
    for _ in range(10):
        for _ in range(substeps):
            state = model.rk4_step(state, -step)
            work["rhs_history_primal"] += 4 * (state.numel() // (5 * model.n))
        values.append(state)
    return torch.stack(values[::-1], dim=-2)


def residual(model, theta, omega, base, du, step, work):
    state = torch.cat((base[..., :2*model.n], base[..., 2*model.n:] + du * model.init_bounds), dim=-1)
    path = history_path(model, state, step, work)
    difference = torch.cat((path[..., :10, :model.n] - theta[..., :10, :],
                            (path[..., :10, model.n:2*model.n] - omega[..., :10, :]) / model.omega_scale), dim=-1)
    return difference.flatten(-2) / math.sqrt(20 * model.n)


def initialize(model, theta, omega, step, work, *, force_fallback=False):
    """Only causal motion and frozen model constants enter the inversion."""
    base = model.initialize(theta, omega).detach()
    work["encoder_events"] += base.numel() // (5 * model.n)
    zero = torch.zeros_like(base[..., 2*model.n:])
    # Each event has its own 3N unknowns; vmap avoids cross-event Jacobian blocks.
    def one(th, om, anchor, du):
        return residual(model, th, om, anchor, du, step, work)
    grouped = base.ndim == 2
    before_jacobian = work["rhs_history_primal"]
    if grouped:
        fn = torch.func.vmap(torch.func.jacfwd(one, argnums=3))
        jac = fn(theta, omega, base, zero)
    else:
        jac = torch.func.jacfwd(one, argnums=3)(theta, omega, base, zero)
    if grouped:
        work["rhs_history_primal"] += (work["rhs_history_primal"]-before_jacobian) * (len(base)-1)
    work["rhs_history_jvp_tangent_equivalents"] += (160 if step == .0025 else 320) * 3 * model.n * (base.numel() // (5*model.n))
    r = residual(model, theta, omega, base, zero, step, work).detach()
    jac = jac.detach()
    batched = zip(base, zero, r, jac, theta, omega) if grouped else [(base, zero, r, jac, theta, omega)]
    states, rows = [], []
    for anchor, z, rr, jj, th, om in batched:
        reason, delta, linear_error, condition = "accepted", z, None, None
        before = float(rr.square().sum())
        after = history_after = None
        try:
            if force_fallback or not (torch.isfinite(rr).all() and torch.isfinite(jj).all()):
                reason = "forced_fallback" if force_fallback else "nonfinite_residual_or_jacobian"
            else:
                a = jj.T @ jj + .002 * torch.eye(z.numel(), dtype=z.dtype, device=z.device)
                b = -(jj.T @ rr)
                condition = float(torch.linalg.cond(a))
                delta = torch.linalg.solve(a, b)
                linear_error = float((a @ delta - b).norm() / b.norm().clamp_min(torch.finfo(b.dtype).tiny))
                trial = residual(model, th, om, anchor, delta, step, work).detach()
                history_after = float(trial.square().sum())
                after = history_after + .001 * float(delta.square().sum())
                if not (torch.isfinite(delta).all() and math.isfinite(after) and math.isfinite(linear_error)):
                    reason = "nonfinite_solve_or_trial"
                elif linear_error > 1e-8:
                    reason = "linear_residual"
                elif not after < before:
                    reason = "objective_not_strictly_lower"
        except torch.linalg.LinAlgError:
            reason = "solve_failure"
        accepted = reason == "accepted"
        state = torch.cat((anchor[:2*model.n], anchor[2*model.n:] + delta * model.init_bounds)) if accepted else anchor.clone()
        states.append(state)
        rows.append({"reason":reason, "accepted":accepted, "before":before, "trial_after":after,
                     "effective_after":after if accepted else before,
                     "trial_history_after":history_after, "effective_history_after":history_after if accepted else before,
                     "linear_residual_relative":linear_error, "solve_condition":condition,
                     "delta_norm":float(delta.norm()) if torch.isfinite(delta).all() else None})
    return (torch.stack(states) if grouped else states[0]), rows


def discrepancy(a, b, scale):
    d = (a-b) / scale
    rms, maximum = float(d.square().mean().sqrt()), float(d.abs().max())
    return {"rms":rms, "maximum":maximum, "passed":math.isfinite(rms) and math.isfinite(maximum) and rms<=1e-4 and maximum<=1e-3}


def qualify(model, events, stored, work):
    indices = [next(i for i,e in enumerate(events) if e["label"]==label) for label in ("stable","unstable")]
    rows = []
    for index in indices:
        event = events[index]
        th, om = [torch.as_tensor(event["history"][k], dtype=torch.float64) for k in ("theta","omega")]
        base = model.initialize(th,om).detach()
        main, reasons = initialize(model, th, om, .0025, work)
        fine, _ = initialize(model, th, om, .00125, work)
        scale = torch.cat((torch.ones(model.n), model.omega_scale, model.init_bounds))
        checks = {"endpoint":bool(torch.equal(main[:2*model.n],base[:2*model.n])),
                  "main_fine":discrepancy(main,fine,scale)}
        shifted, _ = initialize(model,th+.375,om,.0025,work)
        expected = main.clone(); expected[:model.n] += .375
        checks["common_angle_shift"] = discrepancy(shifted,expected,scale)
        back = history_path(model,main,.0025,work)
        forward = back[0]
        for _ in range(40):
            forward = model.rk4_step(forward,.0025)
            work["rhs_history_primal"] += 4
        checks["round_trip"] = discrepancy(forward,main,scale)
        checks["history_main_fine"] = discrepancy(back,history_path(model,fine,.00125,work),scale)
        zero = torch.zeros(3*model.n,dtype=torch.float64)
        fn = lambda u: residual(model,th,om,base,u,.0025,work)
        jac = torch.func.jacfwd(fn)(zero)
        work["rhs_history_jvp_tangent_equivalents"] += 160 * 3 * model.n
        direction = torch.arange(1,3*model.n+1,dtype=torch.float64); direction /= direction.norm()
        ad = jac @ direction
        fdrows = []
        for h in (1e-5,5e-6):
            fd = (fn(h*direction)-fn(-h*direction))/(2*h)
            absolute = float((fd-ad).norm()); magnitude = max(float(fd.norm()),float(ad.norm()))
            fdrows.append({"h":h,"absolute_error":absolute,"relative_error":absolute/magnitude if magnitude else None,
                           "passed":absolute<=1e-6 or absolute<=1e-4*magnitude})
        checks["jacobian_direction"] = {"passed":all(x["passed"] for x in fdrows),"checks":fdrows}
        with torch.no_grad():
            # Only the two frozen qualification prefixes repeat baseline integration.
            prefixes = [("main",.005),("fine",.0025)]
            matches = []
            for name,step in prefixes:
                rebuilt = integrate(model,base,10,internal_step=step)
                work["rhs_baseline_prefix"] += 4 * round(.01/step)*10
                difference = (rebuilt-torch.as_tensor(stored[f"event_{index:04d}_{name}"][:11])) / scale
                maximum = float(difference.abs().max())
                matches.append({"path":name,"normalized_max_absolute_error":maximum,
                                "passed":math.isfinite(maximum) and maximum<=1e-12})
            checks["saved_K0_prefix"] = {"passed":all(x["passed"] for x in matches),"checks":matches}
        rows.append({"event_id":event["id"],"checks":checks,"initialization":reasons,
                     "passed":all(v if isinstance(v,bool) else v["passed"] for v in checks.values())})
    fallback, _ = initialize(model,th,om,.0025,work,force_fallback=True)
    forced = bool(torch.equal(fallback,base))
    return {"passed":forced and all(r["passed"] for r in rows),"forced_fallback_exact":forced,"events":rows}


def evaluate(model, events, constants, directory, work):
    rows, arrays, initial_rows = [], {}, []
    groups = {}
    integration = IntegrationWork()
    costs = {name:{"wall_s":0.,"cpu_s":0.} for name in ("initialization","future_integration_and_scoring")}
    for index,event in enumerate(events):
        steps = max(800,min(1500,len(event["t"])-1)) if event["label"]=="unstable" else 800
        groups.setdefault(steps,[]).append(index)
    for steps,indices in groups.items():
        for start in range(0,len(indices),16):
            batch_indices = indices[start:start+16]
            th,om = [torch.as_tensor(np.stack([events[i]["history"][k] for i in batch_indices]),dtype=torch.float64) for k in ("theta","omega")]
            started = perf_counter(),process_time()
            initial, main_rows = initialize(model,th,om,.0025,work)
            reference, fine_rows = initialize(model,th,om,.00125,work)
            costs["initialization"]["wall_s"] += perf_counter()-started[0]
            costs["initialization"]["cpu_s"] += process_time()-started[1]
            started = perf_counter(),process_time()
            with torch.no_grad():
                main = integrate(model,initial,steps,work=integration)
                fine = integrate(model,reference,steps,internal_step=.0025,work=integration)
                ports = model.ports(fine)
            for local,index in enumerate(batch_indices):
                prefix = f"event_{index:04d}"
                a,b = main[local].numpy(),fine[local].numpy()
                arrays.update({prefix+"_main":a,prefix+"_fine":b,
                               prefix+"_initial_main":initial[local].numpy(),prefix+"_initial_fine":reference[local].numpy()})
                rows.append(score_event(events[index],a,b,{"tau_m":ports.tau_m[local].numpy(),"tau_e":ports.tau_e[local].numpy()},constants))
                initial_rows.append({"event_id":events[index]["id"],"main":main_rows[local],"fine":fine_rows[local]})
            costs["future_integration_and_scoring"]["wall_s"] += perf_counter()-started[0]
            costs["future_integration_and_scoring"]["cpu_s"] += process_time()-started[1]
    by_id = {r["event_id"]:r for r in rows}; rows = [by_id[e["id"]] for e in events]
    np.savez_compressed(directory/"predictions.npz",**arrays)
    refs = {"predictions_ref":old._identity(directory/"predictions.npz"),
            "rows_ref":old._write_json(directory/"rows.json",rows),
            "initialization_ref":old._write_json(directory/"initialization.json",initial_rows)}
    summary = summarize(rows,constants)
    summary["initialization_reasons"] = {path:dict(Counter(r[path]["reason"] for r in initial_rows)) for path in ("main","fine")}
    summary["main_fine_acceptance_disagreement_count"] = sum(r["main"]["accepted"] != r["fine"]["accepted"] for r in initial_rows)
    summary["history_objective"] = {path:{"before_sum":sum(r[path]["before"] for r in initial_rows),
        "effective_after_sum":sum(r[path]["effective_after"] for r in initial_rows),
        "effective_history_after_sum":sum(r[path]["effective_history_after"] for r in initial_rows),
        "accepted_trial_after_sum":sum(r[path]["trial_after"] for r in initial_rows if r[path]["accepted"])} for path in ("main","fine")}
    summary["integration_work"] = integration.snapshot()
    summary["cost"] = costs
    return {"summary":summary,**refs}


def validate(release):
    if (release.get("run_id"),release.get("job_id"),release.get("released")) != (RUN,"probe01",True):
        raise ValueError("fixed released probe01 required")
    if not SOURCES <= release.get("source_hashes",{}).keys():
        raise ValueError("required source hashes missing")
    for name,digest in release["source_hashes"].items():
        if old._identity(old._repo_path(name))["sha256"] != digest:
            raise ValueError("source changed: "+name)
    expected = {"audit_ref":AUDIT,"checkpoint_ref":BASE+"/attempt_1700.pt","parent_pair_ref":PAIR,
                "baseline_rows_ref":BASE+"/evaluation/rows.json","baseline_predictions_ref":BASE+"/evaluation/predictions.npz"}
    for key,path in expected.items():
        if release[key]["path"] != path:
            raise ValueError("noncanonical fixed parent: "+key)
        old._verified_ref(release[key])
    parent = json.loads(old._verified_ref(release["parent_pair_ref"]).read_bytes())
    if (parent.get("status"),parent.get("result",{}).get("fixed_endpoint")) != ("completed",1700):
        raise ValueError("completed P1a receipt required")
    if parent["result"]["treatments"]["8s"]["checkpoint"] != release["checkpoint_ref"] or parent["release"]["audit_ref"] != release["audit_ref"]:
        raise ValueError("P1a checkpoint/audit binding changed")
    return parent


def execute(path):
    started = perf_counter(),process_time()
    path = path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("release must be inside repository")
    release = json.loads(path.read_bytes()); parent = validate(release)
    local,public = [ROOT/side/RUN/"probe01" for side in ("artifacts","results")]
    if local.exists() or public.exists():
        raise FileExistsError("exclusive probe already claimed")
    local.mkdir(parents=True,exist_ok=False)
    receipt = {"run_id":RUN,"job_id":"probe01","release":release,"release_ref":old._identity(path),
               "status":"started","pid":os.getpid(),"started_utc":datetime.now(timezone.utc).isoformat(),"command":[sys.executable,*sys.argv]}
    old._write_json(local/"started.json",receipt)
    try:
        public.mkdir(parents=True,exist_ok=False); torch.set_num_threads(1)
        from kiki.data.osgkd import load_fit297
        events,constants = load_fit297(release["audit_ref"])
        saved = torch.load(old._verified_ref(release["checkpoint_ref"]),map_location="cpu",weights_only=True)
        if saved["attempt"] != 1700 or saved["model"]["config"]["variant"] != "C1" or saved["provenance"]["audit_ref"] != release["audit_ref"]:
            raise ValueError("fixed audited C1 attempt1700 required")
        model = OSGKD.from_checkpoint(saved["model"]).cpu().double().eval().requires_grad_(False)
        if model.n != 10 or not torch.equal(model.init_bounds,torch.tensor([4.]*10+[1.]*10+[4.]*10,dtype=torch.float64)):
            raise ValueError("fixed P1a latent dimension and init_bounds required")
        if len(events)!=297 or sum(e["label"]=="stable" for e in events)!=267:
            raise ValueError("original fit297 denominator required")
        if any(len(e["t"])-1>800 for e in events if e["label"]=="unstable"):
            raise ValueError("released unstable actual horizons exceed the fixed 800-output budget")
        baseline_rows = json.loads(old._verified_ref(release["baseline_rows_ref"]).read_bytes())
        if [r["event_id"] for r in baseline_rows] != [e["id"] for e in events]:
            raise ValueError("baseline frozen event order changed")
        work = Counter()
        with np.load(old._verified_ref(release["baseline_predictions_ref"]),allow_pickle=False) as stored:
            qualification = qualify(model,events,stored,work)
        old._write_json(local/"qualification.json",qualification)
        result = {"passed":qualification["passed"],"qualification":qualification,"evaluation_denominator":297,"parameter_updates":0,"new_fom":0,
                  "baseline_rows_ref":release["baseline_rows_ref"],"baseline_summary_ref":parent["result"]["treatments"]["8s"]["evaluation_summary_ref"]}
        if qualification["passed"]:
            result["candidate"] = evaluate(model,events,constants,local,work)
        result["initialization_work"] = dict(work)
        receipt.update(status="completed",result=result,cost={"wall_s":perf_counter()-started[0],"cpu_s":process_time()-started[1]},
                       environment={"torch":str(torch.__version__),"numpy":np.__version__,"device":"cpu","dtype":"float64","threads":1})
        ref = old._write_json(public/"summary.json",receipt)
        old._write_json(local/"completed.json",{"status":"completed","summary_ref":ref})
        return 0 if qualification["passed"] else 1
    except BaseException as error:
        old._write_json(local/"failed.json",{"status":"failed","error":f"{type(error).__name__}: {error}",
                            "wall_s":perf_counter()-started[0],"cpu_s":process_time()-started[1]})
        raise


def self_check():
    torch.set_num_threads(1); torch.manual_seed(61023)
    constants = {k:np.ones(2) for k in ("mass","damping","Sp","Se","omega_scale","S_m")}
    constants.update(F=np.eye(2),theta_ref=np.zeros(2),edge_index=np.array([[0],[1]]))
    model = old.new_model("C1",constants,torch.device("cpu")).eval().requires_grad_(False)
    theta = torch.linspace(-.02,.03,22,dtype=torch.float64).reshape(11,2)
    omega = torch.full_like(theta,.1); work = Counter()
    base = model.initialize(theta,omega).detach()
    state,rows = initialize(model,theta,omega,.0025,work)
    assert torch.equal(state[:4],base[:4])
    assert all(r["trial_after"]<r["before"] for r in rows if r["accepted"])
    fallback,reasons = initialize(model,theta,omega,.0025,work,force_fallback=True)
    assert torch.equal(fallback,base) and reasons[0]["reason"]=="forced_fallback"
    batch,brows = initialize(model,theta.repeat(2,1,1),omega.repeat(2,1,1),.0025,work)
    assert torch.allclose(batch[0],state,atol=1e-12,rtol=1e-12) and len(brows)==2
    shifted,_ = initialize(model,theta+.375,omega,.0025,work)
    expected = state.clone(); expected[:2] += .375
    assert torch.allclose(shifted,expected,atol=1e-11,rtol=1e-11)
    zero = torch.zeros(6,dtype=torch.float64)
    fn = lambda u: residual(model,theta,omega,base,u,.0025,work)
    direction = torch.arange(1,7,dtype=torch.float64); direction /= direction.norm()
    ad = torch.func.jacfwd(fn)(zero) @ direction
    for h in (1e-5,5e-6):
        fd = (fn(h*direction)-fn(-h*direction))/(2*h)
        assert float((fd-ad).norm()) <= 1e-6
    fine,_ = initialize(model,theta,omega,.00125,work)
    assert discrepancy(state,fine,torch.cat((torch.ones(2),model.omega_scale,model.init_bounds)))["passed"]
    assert all(p.grad is None for p in model.parameters())
    print("synthetic self-check passed: actual AD, grouped solve, fixed endpoints, common angle shift, exact forced fallback, finite differences and fine path; no private assets read")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--self-check",action="store_true")
    group.add_argument("--release",type=Path)
    args = parser.parse_args()
    if args.self_check:
        self_check()
    else:
        raise SystemExit(execute(args.release))
