"""Release-bound P1a qualification and fixed paired continuation; synthetic checks need no data."""
from __future__ import annotations

import argparse
import copy
import ctypes
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
from kiki.data.osgkd import load_fit297
from kiki.models.osgkd import OSGKD
from kiki.models.osgkd_fitting import IntegrationWork, tensor_events, event_objective, joint_loss, integrate

spec = importlib.util.spec_from_file_location("p1a_legacy", ROOT / "experiments/osgkd_fit297_20261113/run.py")
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
RUN = "osgkd_v2_p1a_20261123"
PROTOCOL = "campaigns/osgkd_v2/p1a_protocol.md"
PARENT_ARTIFACTS = "results/osgkd_fit297_20261113/artifacts.json"
SOURCES = {f"experiments/{RUN}/run.py", "experiments/osgkd_fit297_20261113/run.py", PROTOCOL,
           "src/kiki/models/osgkd.py", "src/kiki/models/osgkd_fitting.py", "src/kiki/data/osgkd.py",
           "src/kiki/evaluation/osgkd.py", "experiments/evaluate_l1_latent.py", PARENT_ARTIFACTS}


def memory_observation():
    class Counters(ctypes.Structure):
        _fields_ = [("cb",ctypes.c_ulong),("faults",ctypes.c_ulong)] + [
            (name,ctypes.c_size_t) for name in ("peak_working_set","working_set","peak_paged_pool",
              "paged_pool","peak_nonpaged_pool","nonpaged_pool","pagefile","peak_pagefile")]
    if sys.platform != "win32":
        return {"working_set_bytes":None,"peak_working_set_bytes":None}
    value = Counters()
    value.cb = ctypes.sizeof(value)
    process = ctypes.windll.kernel32.GetCurrentProcess
    process.restype = ctypes.c_void_p
    query = ctypes.windll.psapi.GetProcessMemoryInfo
    query.argtypes = (ctypes.c_void_p,ctypes.POINTER(Counters),ctypes.c_ulong)
    if not query(process(),ctypes.byref(value),value.cb):
        raise ctypes.WinError()
    return {"working_set_bytes":value.working_set,"peak_working_set_bytes":value.peak_working_set}


def restore(saved, count):
    model = OSGKD.from_checkpoint(saved["model"]).cpu().double()
    optimizer = old.new_optimizer(model)
    optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
    batches = old.EventBatches(count)
    state = saved["numpy_batch_rng"]
    batches.rng.bit_generator.state = copy.deepcopy(state["rng"])
    batches.order, batches.position = list(state["order"]), state["position"]
    if sorted(batches.order) != list(range(count)) or not 0 <= batches.position <= count:
        raise ValueError("invalid restored remaining batch order")
    torch.set_rng_state(saved["torch_rng"])
    return model, optimizer, batches


def attempt(model, optimizer, batch, number, steps, scale, work):
    optimizer.zero_grad(set_to_none=True)
    finite = []
    components = event_objective(model,batch,steps,scale,recompute=False,work=work,path_finite=finite)
    loss = joint_loss(components)
    row = {"attempt":number,"horizon_s":steps/100,"event_ids":[e["id"] for e in batch],
           "J":float(loss.detach()),"components":dict(zip(old.COMPONENTS,components.detach().mean(0).tolist(),strict=True)),
           "accepted":False,"clipped":False,"gradient_norm":None,"projection":None}
    if not bool(torch.stack(finite).all()):
        return {**row,"rejection":"nonfinite_trajectory"}
    if not math.isfinite(row["J"]):
        return {**row,"rejection":"nonfinite_loss"}
    loss.backward()
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.0))
    row["gradient_norm"] = norm
    if not math.isfinite(norm):
        return {**row,"rejection":"nonfinite_gradient"}
    optimizer.step()
    with torch.no_grad():
        c,beta = model.c.clone(),model.beta.clone()
        model.project_parameters_()
        change = torch.cat((model.c-c,model.beta-beta))
        values = torch.stack([(change!=0).sum(),change.abs().amax(),(model.c==0).sum(),
                              (model.beta==0).sum(),(model.beta==model.c).sum()]).cpu().tolist()
    count,magnitude,c_zero,beta_zero,beta_cap = values
    return {**row,"accepted":True,"clipped":norm>5.0,
            "projection":{"changed_coordinates":int(count),"max_change":magnitude,
                          "c_zero":int(c_zero),"beta_zero":int(beta_zero),"beta_at_c":int(beta_cap)}}


def fitting_events(events, constants):
    # The old helper fixes fit/check identities; explicit 800-point tensors retain the tail.
    sides, excluded, _ = old.event_sets(events, constants, torch.device("cpu"))
    by_id = {event["id"]: event for event in events}
    for side in sides:
        sides[side] = tensor_events([by_id[e["id"]] for e in sides[side]], "cpu", max_steps=800)
        for event in sides[side]:
            expected = min(801, len(by_id[event["id"]]["t"]))
            if any(len(value) != expected for field in ("targets", "masks") for value in event[field].values()):
                raise ValueError("800-step target coverage changed")
    return sides, excluded


def frozen_sequence(saved, fit):
    _, _, batches = restore(saved, len(fit))
    return [[fit[i]["id"] for i in batches.next()] for _ in range(200)]


def objective(model, events, scale, steps=800, work=None):
    return joint_loss(event_objective(model, events, steps, scale, recompute=False,work=work))


@torch.no_grad()
def numerical(model, events, work):
    from kiki.evaluation.osgkd import numerical_gate

    rows = []
    for event in events:
        initial = model.initialize(event["history"]["theta"], event["history"]["omega"])
        steps = min(800, len(event["targets"]["theta"]) - 1)
        work.encoder_events += 1
        main = integrate(model, initial, steps, recompute=False,work=work)
        fine = integrate(model, initial, steps, internal_step=.0025, recompute=False,work=work)
        gate = numerical_gate(main.numpy(), fine.numpy(), model.omega_scale.numpy())
        rows.append({"id": event["id"], "steps": steps, **gate})
    return {"passed": all(row["passed"] for row in rows), "events": rows}


def directional(model, event, scale, work):
    groups = old.parameter_groups(model)
    generator = torch.Generator().manual_seed(old.SEED)
    rows = []
    for name, group in zip(("encoder", "dynamics", "edges"), groups, strict=True):
        vector = torch.randn(sum(p.numel() for p in group), generator=generator, dtype=torch.float64)
        if name == "edges":
            c, beta = (parameter.detach() for parameter in group)
            interior = (beta > 2e-5) & (c - beta > 2e-5)
            if not interior.any():
                rows.append({"direction": name, "status": "no_interior_coordinates", "passed": None})
                continue
            vector *= interior.repeat(2)
        vector /= vector.norm()
        direction = [v.reshape_as(p) for v,p in zip(vector.split([p.numel() for p in group]),group,strict=True)]
        if name == "edges" and not old._fd_edge_feasible(model, direction):
            raise ValueError("fixed edge direction left the feasible interior")
        loss = objective(model, [event], scale,work=work)
        gradients = torch.autograd.grad(loss, group, allow_unused=True)
        ad = sum(float((g*v).sum()) for g,v in zip(gradients,direction,strict=True) if g is not None)
        original = [p.detach().clone() for p in group]
        with torch.no_grad():
            try:
                for h in (1e-5,5e-6):
                    values = []
                    for sign in (-1,1):
                        for p,x,v in zip(group,original,direction,strict=True):
                            p.copy_(x+sign*h*v)
                        values.append(float(objective(model,[event],scale,work=work)))
                    fd = (values[1]-values[0])/(2*h)
                    error = abs(fd-ad)
                    limit = 1e-8 + 1e-3*max(abs(ad),abs(fd))
                    rows.append({"direction":name,"h":h,"ad":ad,"fd":fd,"absolute_error":error,
                                 "tolerance":limit,"passed":math.isfinite(error) and error<=limit})
            finally:
                for p,x in zip(group,original,strict=True):
                    p.copy_(x)
    return {"passed": all(row["passed"] is not False for row in rows),
            "checks": rows, "event_id": event["id"]}


def grouped_check(model, events, scale, steps=800, work=None):
    parameters = list(model.parameters())
    started = perf_counter(),process_time()
    grouped = objective(model,events,scale,steps,work)
    ga = torch.autograd.grad(grouped, parameters, materialize_grads=True)
    grouped_value = float(grouped.detach())
    grouped_cost = {"wall_s":perf_counter()-started[0],"cpu_s":process_time()-started[1]}
    started = perf_counter(),process_time()
    gb = [torch.zeros_like(parameter) for parameter in parameters]
    sequential_value = 0.0
    for event in events:
        loss = objective(model, [event], scale, steps, work)
        gradient = torch.autograd.grad(loss, parameters, materialize_grads=True)
        sequential_value += float(loss.detach()) / len(events)
        for total, value in zip(gb, gradient, strict=True):
            total.add_(value.detach(), alpha=1 / len(events))
    sequential_cost = {"wall_s":perf_counter()-started[0],"cpu_s":process_time()-started[1]}
    gradient_error = max(float((a-b).abs().max()) for a,b in zip(ga,gb,strict=True))
    magnitude = max(float(a.abs().max()) for a in ga)
    loss_error = abs(grouped_value - sequential_value)
    return {"event_ids":[e["id"] for e in events],"loss_absolute_error":loss_error,
            "gradient_max_absolute_error":gradient_error,"gradient_tolerance":1e-10+1e-8*magnitude,
            "grouped_cost":grouped_cost,"sequential_cost":sequential_cost,
            "passed": math.isfinite(loss_error) and math.isfinite(gradient_error)
            and loss_error <= 1e-10+1e-8*abs(grouped_value) and gradient_error <= 1e-10+1e-8*magnitude}


def qualify(saved, fit, raw, constants, directory):
    work = IntegrationWork()
    sequence = frozen_sequence(saved,fit)
    sequence_ref = old._write_json(directory/"batch_sequence.json",sequence)
    trials = []
    for _ in range(2):
        model,optimizer,batches = restore(saved,len(fit))
        batch = [fit[i] for i in batches.next()]
        row = attempt(model,optimizer,batch,1501,200,constants["S_m"],work)
        trials.append((row,model.checkpoint()["state_dict"],copy.deepcopy(optimizer.state_dict()),torch.get_rng_state()))
    repeat = (trials[0][0] == trials[1][0] and tree_equal(trials[0][1:],trials[1][1:]))
    model,_,_ = restore(saved,len(fit))
    labels = {e["id"]:e["label"] for e in raw}
    ordered_ids = list(dict.fromkeys(event_id for batch in sequence for event_id in batch))
    ordered = sorted(fit,key=lambda e:ordered_ids.index(e["id"]))
    long = [e for e in ordered if len(e["targets"]["theta"])==801 and labels[e["id"]]=="stable"]
    short = [e for e in ordered if len(e["targets"]["theta"])<801 and labels[e["id"]]=="unstable"]
    if not long or not short:
        raise ValueError("qualification requires long stable and short true futures")
    numeric = numerical(model,[long[0],short[0]],work)
    gradient = directional(model,long[0],constants["S_m"],work)
    grouped = grouped_check(model,long[:8],constants["S_m"],work=work)
    return {"passed":repeat and trials[0][0]["accepted"] and numeric["passed"] and gradient["passed"] and grouped["passed"],
            "restored_trial_equal":repeat,"trial":trials[0][0],"numerical":numeric,"gradient":gradient,
            "grouped":grouped,"batch_sequence_ref":sequence_ref,"trial_attempts":len(trials),
            "discarded_trial_updates":sum(trial[0]["accepted"] for trial in trials),
            "work":old._work_receipt(work,with_backward=True),"memory_observation":memory_observation()}


def tree_equal(a,b):
    if isinstance(a,torch.Tensor):
        return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):
        return a.keys()==b.keys() and all(tree_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):
        return len(a)==len(b) and all(tree_equal(x,y) for x,y in zip(a,b,strict=True))
    return a==b


def compare(saved, fit, events, constants, directory, release, qualification):
    from kiki.evaluation.osgkd import evaluate_fit297
    sequence = json.loads(old._verified_ref(qualification["result"]["batch_sequence_ref"]).read_bytes())
    if sequence != frozen_sequence(saved,fit):
        raise ValueError("qualification batch order differs")
    outputs,states = {},{}
    for steps in (200,800):
        name = f"{steps//100}s"
        target = directory/name
        target.mkdir()
        (target/"attempts.jsonl").touch(exist_ok=False)
        model,optimizer,batches = restore(saved,len(fit))
        states[steps] = {"model":model,"optimizer":optimizer,"batches":batches,"work":IntegrationWork(),
                         "rows":[],"rejected":0,"consecutive":0,"target":target,"rng":torch.get_rng_state(),
                         "wall_s":0.0,"cpu_s":0.0}
    for offset,expected in enumerate(sequence,1501):
        stopped = False
        for steps,state in states.items():
            model,optimizer,batches = (state[key] for key in ("model","optimizer","batches"))
            torch.set_rng_state(state["rng"])
            batch = [fit[i] for i in batches.next()]
            if [e["id"] for e in batch] != expected:
                raise ValueError("paired order changed")
            wall, cpu = perf_counter(), process_time()
            row = attempt(model,optimizer,batch,offset,steps,constants["S_m"],state["work"])
            row.update(wall_s=perf_counter()-wall, cpu_s=process_time()-cpu)
            state["wall_s"] += row["wall_s"]
            state["cpu_s"] += row["cpu_s"]
            with (state["target"]/"attempts.jsonl").open("a",encoding="utf-8",newline="\n") as log:
                log.write(json.dumps(old._json_safe(row),allow_nan=False)+"\n")
                log.flush()
            state["rng"] = torch.get_rng_state()
            state["rows"].append(row)
            state["consecutive"] = 0 if row["accepted"] else state["consecutive"]+1
            state["rejected"] += not row["accepted"]
            stopped |= state["consecutive"]>=10 or state["rejected"]>=30
        if stopped:
            for state in states.values():
                old._write_json(state["target"]/"attempts.json",state["rows"])
            raise RuntimeError("old nonfinite rejection stop; both treatments stopped at same paired batch")
    for steps,state in states.items():
        name = f"{steps//100}s"
        model,optimizer,batches,work,rows,rejected,target = (state[key] for key in (
            "model","optimizer","batches","work","rows","rejected","target"))
        torch.set_rng_state(state["rng"])
        old._write_json(target/"attempts.json",rows)
        checkpoint = old._checkpoint(target/"attempt_1700.pt",model,optimizer,batches,1700,
                                     {"audit_ref":release["audit_ref"],"release":release},torch.device("cpu"))
        outputs[name] = {"checkpoint":checkpoint,"attempts":200,"accepted":200-rejected,"rejected":rejected,
                         "clipped":sum(r["clipped"] for r in rows),"work":old._work_receipt(work,with_backward=True),
                         "train_cost":{"wall_s":state["wall_s"],"cpu_s":state["cpu_s"]}}
    for steps,state in states.items():
        target = state["target"]
        evaluation_dir = target/"evaluation"
        evaluation_dir.mkdir()
        with torch.no_grad():
            scores = evaluate_fit297(state["model"].eval(),events,constants,output_directory=evaluation_dir)
        outputs[f"{steps//100}s"].update(
            evaluation_cost=scores["cost"],
            evaluation_summary_ref=old._identity(evaluation_dir/"summary.json"),
        )
    return {"treatments":outputs,"paired_batch_sequence_ref":qualification["result"]["batch_sequence_ref"],
            "fixed_endpoint":1700,"new_fom":0}


def validate(release,phase):
    if (release.get("run_id"),release.get("job_id"),release.get("phase"),release.get("released"),
        release.get("device"),release.get("threads"),release.get("recompute")) != (
            RUN,"qual01" if phase=="qualify" else "pair01",phase,True,"cpu",1,False):
        raise ValueError("release is outside P1a job contract")
    if not SOURCES<=release.get("source_hashes",{}).keys():
        raise ValueError("required source hash missing")
    for name,digest in release["source_hashes"].items():
        if old._identity(old._repo_path(name))["sha256"]!=digest:
            raise ValueError("source identity changed: "+name)
    parent = json.loads((ROOT/PARENT_ARTIFACTS).read_bytes())
    if (release["audit_ref"]["path"] != "results/osgkd_fit297_20261113/port_audit_release.json"
            or release["checkpoint_ref"] != parent["selected_models"]["C1_dynamic"]):
        raise ValueError("the fixed C1 parent and fit297 audit release are required")
    old._verified_ref(release["audit_ref"])
    old._verified_ref(release["checkpoint_ref"])
    qualified = None
    if phase=="compare":
        if release["qualification_ref"]["path"] != f"results/{RUN}/qual01/summary.json":
            raise ValueError("the passing qualification must belong to this run")
        qualified = json.loads(old._verified_ref(release["qualification_ref"]).read_bytes())
        if (qualified.get("run_id"),qualified.get("job_id"),qualified.get("phase"),qualified.get("status"),
            qualified.get("result",{}).get("passed")) != (RUN,"qual01","qualify","completed",True):
            raise ValueError("passing P1a qualification required")
        for key in ("audit_ref","checkpoint_ref","source_hashes","device","threads","recompute"):
            if qualified["release"][key]!=release[key]:
                raise ValueError("qualification inputs differ: "+key)
        old._verified_ref(qualified["result"]["batch_sequence_ref"])
    return qualified


def execute(phase,path):
    started = perf_counter(),process_time()
    path = path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("release must be inside repository")
    release = json.loads(path.read_bytes())
    qualified = validate(release,phase)
    local,public = (ROOT/side/RUN/release["job_id"] for side in ("artifacts","results"))
    if local.exists() or public.exists():
        raise FileExistsError("exclusive job already claimed")
    local.mkdir(parents=True,exist_ok=False)
    receipt = {"run_id":RUN,"job_id":release["job_id"],"phase":phase,"release":release,
               "release_ref":old._identity(path),"status":"started","pid":os.getpid(),
               "started_utc":datetime.now(timezone.utc).isoformat(),"command":[sys.executable,*sys.argv]}
    old._write_json(local/"started.json",receipt)
    try:
        public.mkdir(parents=True,exist_ok=False)
        torch.set_num_threads(1)
        events,constants = load_fit297(release["audit_ref"])
        saved = torch.load(old._verified_ref(release["checkpoint_ref"]),map_location="cpu",weights_only=True)
        if saved["attempt"]!=1500 or saved["model"]["config"]["variant"]!="C1" or saved["provenance"]["audit_ref"]!=release["audit_ref"]:
            raise ValueError("base must be the audited C1 attempt 1500")
        if len(events)!=297 or sum(e["label"]=="unstable" for e in events)!=30:
            raise ValueError("exact fit297 cohort required")
        sides,excluded = fitting_events(events,constants)
        result = (qualify(saved,sides["fit"],events,constants,local) if phase=="qualify" else
                  compare(saved,sides["fit"],events,constants,local,release,qualified))
        result["objective_excluded_ids"] = excluded
        result["evaluation_denominator"] = 297
        receipt.update(status="completed",result=result)
        receipt["cost"] = {"wall_s":perf_counter()-started[0],"cpu_s":process_time()-started[1]}
        receipt["environment"] = {"torch":str(torch.__version__),"numpy":np.__version__,"device":"cpu",
                                  "dtype":"float64","threads":torch.get_num_threads(),"recompute":False}
        summary = old._write_json(public/"summary.json",receipt)
        old._write_json(local/"completed.json",{"status":"completed","summary_ref":summary})
        return 0 if result.get("passed",True) else 1
    except BaseException as error:
        old._write_json(local/"failed.json",{"status":"failed","error":f"{type(error).__name__}: {error}",
                            "wall_s":perf_counter()-started[0],"cpu_s":process_time()-started[1]})
        raise


def self_check():
    torch.set_num_threads(1)
    constants = {key:np.ones(2) for key in ("mass","damping","Sp","Se","omega_scale","S_m")}
    constants.update(F=np.eye(2),theta_ref=np.zeros(2),edge_index=np.array([[0],[1]]))
    model = old.new_model("C1",constants,torch.device("cpu"))
    def event(name,count):
        return {"id":name,"history":{k:np.zeros((11,2)) for k in ("theta","omega")},
                "t":np.arange(count+1)*.01,"targets":{k:np.zeros((count+1,2)) for k in ("theta","omega","tau_m","tau_e")},
                "masks":{k:np.ones((count+1,2),dtype=bool) for k in ("theta","omega","tau_m","tau_e")}}
    raw = [event("long",900),event("short",7)]
    data = tensor_events(raw,"cpu",max_steps=800)
    assert [len(e["targets"]["theta"]) for e in data]==[801,8]
    altered = copy.deepcopy(data)
    altered[0]["targets"]["theta"][201:]+=2
    with torch.no_grad():
        base2 = objective(model,[data[0]],constants["S_m"],200)
        changed2 = objective(model,[altered[0]],constants["S_m"],200)
        base8 = objective(model,[data[0]],constants["S_m"],800)
        changed8 = objective(model,[altered[0]],constants["S_m"],800)
    assert torch.equal(base2,changed2) and abs(float(base8-changed8))>1e-3
    assert grouped_check(model,tensor_events([event("a",12),event("b",12)],"cpu",max_steps=800),constants["S_m"],12)["passed"]
    assert event_objective(model,[data[1]],800,constants["S_m"]).shape==(1,3)
    batches = old.EventBatches(2)
    batches.next()
    saved = {"model":model.checkpoint(),"optimizer":old.new_optimizer(model).state_dict(),
             "numpy_batch_rng":batches.state(),"torch_rng":torch.get_rng_state()}
    assert frozen_sequence(saved,data)==frozen_sequence(saved,data)
    a = restore(saved,2)
    rowa = attempt(a[0],a[1],[data[1]],1501,200,constants["S_m"],IntegrationWork())
    b = restore(saved,2)
    rowb = attempt(b[0],b[1],[data[1]],1501,200,constants["S_m"],IntegrationWork())
    assert rowa==rowb and tree_equal(a[0].state_dict(),b[0].state_dict())
    c = restore(saved,2)
    original = old.training_attempt(c[0],c[1],[data[1]],1501,constants["S_m"],IntegrationWork())
    assert rowa==original and tree_equal(a[0].state_dict(),c[0].state_dict())
    assert tree_equal(a[1].state_dict(),c[1].state_dict())
    assert memory_observation()["peak_working_set_bytes"] is not None
    print("synthetic self-check passed: explicit 800 targets, tail loss, short futures, grouped gradients, restored updates; no private data opened")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("qualify","compare","self-check"))
    parser.add_argument("--release",type=Path)
    args = parser.parse_args()
    if args.phase=="self-check":
        self_check()
    else:
        if args.release is None:
            parser.error("--release is required")
        raise SystemExit(execute(args.phase,args.release))
