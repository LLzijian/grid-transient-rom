"""Release-bound electrical auxiliary qualification and 32 matched continuation attempts."""
from __future__ import annotations

import argparse
import copy
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
from kiki.models.osgkd_fitting import IntegrationWork, integrate, event_objective, joint_loss, tensor_events

spec = importlib.util.spec_from_file_location("electrical_p1a", ROOT / "experiments/osgkd_v2_p1a_20261123/run.py")
p1a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p1a)
old = p1a.old
RUN = "osgkd_v2_early_electrical_20261123"
SOURCES = (p1a.SOURCES - {p1a.PROTOCOL}) | {f"experiments/{RUN}/run.py", f"experiments/{RUN}/analyze.py",
                                                        "campaigns/osgkd_v2/early_electrical_protocol.md"}
PAIR = "results/osgkd_v2_p1a_20261123/pair01/summary.json"
CLOSURE = "results/osgkd_v2_early_closure_20261123/probe01/summary.json"


def require(value, message):
    if not value:
        raise ValueError(message)


def electrical_loss(model, events, work, *, full=False):
    """Average masked electrical MSE equally over events, excluding the initial point."""
    work.encoder_events += len(events)
    initial = model.initialize(torch.stack([e["history"]["theta"] for e in events]),
                               torch.stack([e["history"]["omega"] for e in events]))
    losses, paths = [], []
    for index, event in enumerate(events):
        steps = min(25, len(event["targets"]["theta"])-1)
        require(steps >= 1, "future unavailable")
        path = integrate(model, initial[index], min(800, len(event["targets"]["theta"])-1) if full else steps,
                         recompute=False, work=work)[:steps+1]
        prediction = model.ports(path[1:]).tau_e
        mask = event["masks"]["tau_e"][1:steps+1]
        require(bool(mask.any()), "early electrical labels unavailable: "+event["id"])
        error = (prediction-event["targets"]["tau_e"][1:steps+1])/model.Se
        losses.append(error.square()[mask].mean())
        paths.append(path)
    return torch.stack(losses).mean(), paths


def attempt(model, optimizer, batch, number, scale, work, coefficient):
    if coefficient == 0:
        row = p1a.attempt(model, optimizer, batch, number, 800, scale, work)
        return {**row, "auxiliary_coefficient": 0., "auxiliary_mse": None}
    optimizer.zero_grad(set_to_none=True)
    finite = []
    components = event_objective(model, batch, 800, scale, recompute=False, work=work, path_finite=finite)
    base = joint_loss(components)
    aux, paths = electrical_loss(model, batch, work)
    loss = base+coefficient*aux
    row = {"attempt": number, "horizon_s": 8., "event_ids": [e["id"] for e in batch],
           "J": float(loss.detach()), "base_J": float(base.detach()), "auxiliary_mse": float(aux.detach()),
           "auxiliary_coefficient": coefficient,
           "components": dict(zip(old.COMPONENTS, components.detach().mean(0).tolist(), strict=True)),
           "accepted": False, "clipped": False, "gradient_norm": None, "projection": None}
    if not bool(torch.stack(finite).all()) or not all(bool(torch.isfinite(p).all()) for p in paths):
        return {**row, "rejection": "nonfinite_trajectory"}
    if not math.isfinite(row["J"]):
        return {**row, "rejection": "nonfinite_loss"}
    loss.backward()
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.))
    row["gradient_norm"] = norm
    if not math.isfinite(norm):
        return {**row, "rejection": "nonfinite_gradient"}
    optimizer.step()
    with torch.no_grad():
        c, beta = model.c.clone(), model.beta.clone()
        model.project_parameters_()
        change = torch.cat((model.c-c, model.beta-beta))
        count, magnitude, cz, bz, bc = torch.stack([(change != 0).sum(), change.abs().amax(),
            (model.c == 0).sum(), (model.beta == 0).sum(), (model.beta == model.c).sum()]).tolist()
    return {**row, "accepted": True, "clipped": norm > 5,
            "projection": {"changed_coordinates": int(count), "max_change": magnitude,
                           "c_zero": int(cz), "beta_zero": int(bz), "beta_at_c": int(bc)}}


def sequence(saved, fit):
    _, _, batches = p1a.restore(saved, len(fit))
    return [[fit[i]["id"] for i in batches.next()] for _ in range(32)]


def equivalence(model, event, work):
    parameters = list(model.parameters())
    a, pa = electrical_loss(model, [event], work)
    ga = torch.autograd.grad(a, parameters, materialize_grads=True)
    b, pb = electrical_loss(model, [event], work, full=True)
    gb = torch.autograd.grad(b, parameters, materialize_grads=True)
    checks = [torch.allclose(a, b, atol=1e-10, rtol=1e-8),
              torch.allclose(pa[0], pb[0], atol=1e-10, rtol=1e-8)]
    checks += [torch.allclose(x,y,atol=1e-10,rtol=1e-8) for x,y in zip(ga,gb,strict=True)]
    return {"event_id": event["id"], "passed": all(checks), "loss_absolute_error": abs(float((a-b).detach())),
            "state_max_absolute_error": float((pa[0]-pb[0]).detach().abs().max()),
            "gradient_max_absolute_error": max(float((x-y).abs().max()) for x,y in zip(ga,gb,strict=True))}


def directional(model, event, work):
    generator = torch.Generator().manual_seed(old.SEED)
    rows = []
    for name, group in zip(("encoder", "non_edge_dynamics"), old.parameter_groups(model)[:2], strict=True):
        vector = torch.randn(sum(p.numel() for p in group), generator=generator, dtype=torch.float64)
        vector /= vector.norm()
        direction = [x.reshape_as(p) for x,p in zip(vector.split([p.numel() for p in group]),group,strict=True)]
        loss, _ = electrical_loss(model, [event], work)
        gradients = torch.autograd.grad(loss,group,allow_unused=True)
        ad = sum(float((g*v).sum()) for g,v in zip(gradients,direction,strict=True) if g is not None)
        originals = [p.detach().clone() for p in group]
        with torch.no_grad():
            try:
                for eps in (1e-5,5e-6):
                    values = []
                    for sign in (-1,1):
                        for p,x,v in zip(group,originals,direction,strict=True):
                            p.copy_(x+sign*eps*v)
                        values.append(float(electrical_loss(model,[event],work)[0]))
                    fd = (values[1]-values[0])/(2*eps)
                    denominator = max(abs(fd),abs(ad),1e-8)
                    relative_error = abs(fd-ad)/denominator
                    tolerance = 1e-3*denominator
                    rows.append({"direction":name,"eps":eps,"ad":ad,"fd":fd,"tolerance":tolerance,
                                 "relative_error":relative_error,
                                 "passed":math.isfinite(fd) and relative_error<=1e-3})
            finally:
                for p,x in zip(group,originals,strict=True):
                    p.copy_(x)
    return {"passed":all(r["passed"] for r in rows),"checks":rows,"event_id":event["id"]}


def qualify(saved, fit, raw, constants, directory):
    for event in fit:
        require(bool(event["masks"]["tau_e"][1:26].any()),
                "early electrical labels unavailable: "+event["id"])
    work = IntegrationWork()
    order = sequence(saved,fit)
    ref = old._write_json(directory/"batch_sequence.json",order)
    labels = {e["id"]:e["label"] for e in raw}
    ordered = list(dict.fromkeys(x for batch in order for x in batch))
    lookup = {e["id"]:e for e in fit}
    stable = next(lookup[x] for x in ordered if labels[x]=="stable" and len(lookup[x]["targets"]["theta"])==801)
    unstable = next(lookup[x] for x in ordered if labels[x]=="unstable" and
                    bool(lookup[x]["masks"]["tau_e"][1:26].any()))
    model,_,_ = p1a.restore(saved,len(fit))
    checks = [equivalence(model,e,work) for e in (stable,unstable)]
    fd = [directional(model,e,work) for e in (stable,unstable)]
    trials = []
    for _ in range(2):
        model,opt,batches = p1a.restore(saved,len(fit))
        row = attempt(model,opt,[fit[i] for i in batches.next()],1701,constants["S_m"],work,0.)
        trials.append((row,model.checkpoint(),copy.deepcopy(opt.state_dict()),torch.get_rng_state()))
    repeat = p1a.tree_equal(trials[0],trials[1])
    return {"passed":repeat and trials[0][0]["accepted"] and all(x["passed"] for x in checks+fd),
            "batch_sequence_ref":ref,"equivalence":checks,"directional":fd,"trial":trials[0][0],
            "restored_trial_equal":repeat,"discarded_trial_updates":sum(t[0]["accepted"] for t in trials),
            "work":old._work_receipt(work,with_backward=True)}


def validate(release):
    require((release.get("run_id"),release.get("job_id"),release.get("released"),release.get("device"),
             release.get("threads"),release.get("recompute")) == (RUN,"pair01",True,"cpu",1,False), "release contract differs")
    require(SOURCES <= release.get("source_hashes",{}).keys(), "missing source hash")
    for name,digest in release["source_hashes"].items():
        require(old._identity(old._repo_path(name))["sha256"]==digest,"source identity differs: "+name)
    for key,path in (("parent_pair_ref",PAIR),("early_closure_ref",CLOSURE),
                     ("audit_ref","results/osgkd_fit297_20261113/port_audit_release.json")):
        require(release[key]["path"]==path,"noncanonical reference: "+key)
        require(set(release[key])=={"path","bytes","sha256"},"typed reference required")
        old._verified_ref(release[key])
    parent = json.loads(old._verified_ref(release["parent_pair_ref"]).read_bytes())
    closure = json.loads(old._verified_ref(release["early_closure_ref"]).read_bytes())
    require(parent["status"]=="completed" and parent["result"]["fixed_endpoint"]==1700,"closed P1a pair required")
    require(closure["status"]=="completed" and closure["result"]["gate"]["selected"]=="A_e","A_e gate required")
    require(closure["release"]["parent_pair_ref"]==release["parent_pair_ref"],"closure parent differs")
    require(parent["release"]["audit_ref"]==release["audit_ref"]==closure["release"]["audit_ref"],"audit binding differs")
    require(release["checkpoint_ref"]==parent["result"]["treatments"]["8s"]["checkpoint"],"fixed long checkpoint required")
    require(release["checkpoint_ref"]["path"]=="artifacts/osgkd_v2_p1a_20261123/pair01/8s/attempt_1700.pt","checkpoint path differs")
    old._verified_ref(release["checkpoint_ref"])


def compare(saved, fit, raw, constants, directory, release, qualification):
    from kiki.evaluation.osgkd import evaluate_fit297
    order = json.loads(old._verified_ref(qualification["batch_sequence_ref"]).read_bytes())
    require(order==sequence(saved,fit),"frozen sequence differs")
    states = {}
    for name,coefficient in (("control",0.),("candidate",.1)):
        model,opt,batches = p1a.restore(saved,len(fit))
        target = directory/name
        target.mkdir()
        states[name] = dict(model=model,opt=opt,batches=batches,coefficient=coefficient,target=target,
                            work=IntegrationWork(),rng=torch.get_rng_state(),rows=[],consecutive=0,rejected=0,wall_s=0.,cpu_s=0.)
    stopped = False
    recovery_refs = []
    for number,expected in enumerate(order,1701):
        for state in states.values():
            torch.set_rng_state(state["rng"])
            batch = [fit[i] for i in state["batches"].next()]
            require([e["id"] for e in batch]==expected,"paired order differs")
            start,cpu = perf_counter(),process_time()
            row = attempt(state["model"],state["opt"],batch,number,constants["S_m"],state["work"],state["coefficient"])
            row.update(wall_s=perf_counter()-start,cpu_s=process_time()-cpu)
            state["wall_s"] += row["wall_s"]; state["cpu_s"] += row["cpu_s"]
            state["rng"] = torch.get_rng_state()
            state["rows"].append(row)
            with (state["target"]/"attempts.jsonl").open("a",encoding="utf-8",newline="\n") as stream:
                stream.write(json.dumps(old._json_safe(row),allow_nan=False)+"\n"); stream.flush(); os.fsync(stream.fileno())
            state["consecutive"] = 0 if row["accepted"] else state["consecutive"]+1
            state["rejected"] += not row["accepted"]
            stopped |= state["consecutive"]>=3 or state["rejected"]>=5
        if stopped:
            break
        if number in (1708,1716,1724):
            recovery = directory/"recovery"/f"attempt_{number}"
            recovery.mkdir(parents=True,exist_ok=False)
            snapshots = {}
            for name,state in states.items():
                torch.set_rng_state(state["rng"])
                snapshots[name] = old._checkpoint(recovery/f"{name}.pt",state["model"],state["opt"],
                    state["batches"],number,{"audit_ref":release["audit_ref"],"release":release,
                                           "recovery_only":True},torch.device("cpu"))
            recovery_refs.append(old._write_json(recovery/"manifest.json",{
                "run_id":RUN,"job_id":"pair01","attempt":number,"recovery_only":True,
                "completed_pairs":number-1700,"checkpoints":snapshots,
                "completed_utc":datetime.now(timezone.utc).isoformat()}))
    outputs = {}
    for name,state in states.items():
        torch.set_rng_state(state["rng"])
        checkpoint = old._checkpoint(state["target"]/f"attempt_{number}.pt",state["model"],state["opt"],state["batches"],number,
                                     {"audit_ref":release["audit_ref"],"release":release},torch.device("cpu"))
        outputs[name] = {"checkpoint":checkpoint,"attempts":len(state["rows"]),"accepted":len(state["rows"])-state["rejected"],
                         "rejected":state["rejected"],"clipped":sum(r["clipped"] for r in state["rows"]),
                         "work":old._work_receipt(state["work"],with_backward=True),
                         "attempts_ref":old._write_json(state["target"]/"attempts.json",state["rows"]),
                         "train_cost":{"wall_s":state["wall_s"],"cpu_s":state["cpu_s"]}}
    rhs = sum(s["work"].rhs["forward"] for s in states.values())
    require(rhs<=3328000,"paired training RHS cap exceeded")
    if not stopped:
        for name,state in states.items():
            target = state["target"]/"evaluation"; target.mkdir()
            with torch.no_grad():
                scores = evaluate_fit297(state["model"].eval(),raw,constants,output_directory=target)
            outputs[name].update(evaluation_cost=scores["cost"],evaluation_summary_ref=old._identity(target/"summary.json"))
    return {"treatments":outputs,"target_endpoint":1732,"actual_endpoint":number,"fixed_endpoint":1732 if not stopped else None,
            "stopped_early":stopped,"recovery_manifest_refs":recovery_refs,
            "training_rhs_forward":rhs,"training_rhs_cap":3328000,"new_fom":0}


def execute(path):
    start,cpu = perf_counter(),process_time()
    path = path.resolve(); require(path.is_relative_to(ROOT),"release must be inside repository")
    release = json.loads(path.read_bytes()); validate(release)
    local,public = [ROOT/side/RUN/"pair01" for side in ("artifacts","results")]
    require(not local.exists() and not public.exists(),"exclusive job already claimed")
    local.mkdir(parents=True,exist_ok=False)
    receipt = {"run_id":RUN,"job_id":"pair01","release":release,"release_ref":old._identity(path),"status":"started",
               "pid":os.getpid(),"started_utc":datetime.now(timezone.utc).isoformat(),"command":[sys.executable,*sys.argv]}
    old._write_json(local/"started.json",receipt)
    try:
        public.mkdir(parents=True,exist_ok=False); torch.set_num_threads(1)
        self_check()
        raw,constants = p1a.load_fit297(release["audit_ref"])
        require(len(raw)==297 and sum(e["label"]=="unstable" for e in raw)==30,"fit297 denominator differs")
        saved = torch.load(old._verified_ref(release["checkpoint_ref"]),map_location="cpu",weights_only=True)
        require(saved["attempt"]==1700 and saved["model"]["config"]["variant"]=="C1" and
                saved["provenance"]["audit_ref"]==release["audit_ref"],"audited C1 attempt1700 required")
        sides,excluded = p1a.fitting_events(raw,constants)
        qdir = local/"qualification"; qdir.mkdir()
        qualification = qualify(saved,sides["fit"],raw,constants,qdir)
        qref = old._write_json(qdir/"qualification.json",qualification)
        require(qualification["passed"],"qualification gate failed; trial updates discarded")
        result = compare(saved,sides["fit"],raw,constants,local,release,qualification)
        result.update(qualification=qualification,qualification_ref=qref,objective_excluded_ids=excluded,evaluation_denominator=297)
        receipt.update(status="completed",completed_utc=datetime.now(timezone.utc).isoformat(),result=result,
                       cost={"wall_s":perf_counter()-start,"cpu_s":process_time()-cpu},
                       environment={"torch":str(torch.__version__),"numpy":np.__version__,"device":"cpu","dtype":"float64","threads":1,"recompute":False},
                       memory_observation=p1a.memory_observation())
        ref = old._write_json(public/"summary.json",receipt)
        old._write_json(local/"completed.json",{"status":"completed","summary_ref":ref})
        return 0
    except BaseException as error:
        old._write_json(local/"failed.json",{"status":"failed","error":f"{type(error).__name__}: {error}",
            "failed_utc":datetime.now(timezone.utc).isoformat(),"wall_s":perf_counter()-start,"cpu_s":process_time()-cpu})
        raise


def self_check():
    torch.set_num_threads(1)
    constants = {key:np.ones(2) for key in ("mass","damping","Sp","Se","omega_scale","S_m")}
    constants.update(F=np.eye(2),theta_ref=np.zeros(2),edge_index=np.array([[0],[1]]))
    model = old.new_model("C1",constants,torch.device("cpu"))
    def event(name,count):
        return {"id":name,"history":{k:np.zeros((11,2)) for k in ("theta","omega")},"t":np.arange(count+1)*.01,
                "targets":{k:np.zeros((count+1,2)) for k in ("theta","omega","tau_m","tau_e")},
                "masks":{k:np.ones((count+1,2),bool) for k in ("theta","omega","tau_m","tau_e")}}
    data = tensor_events([event("long",35),event("short",3)],"cpu",max_steps=800)
    work = IntegrationWork()
    base = electrical_loss(model,data,work)[0]
    changed = copy.deepcopy(data)
    changed[0]["targets"]["tau_e"][0] = 1e9
    changed[0]["targets"]["tau_e"][26:] = 1e9
    assert torch.equal(base,electrical_loss(model,changed,work)[0])
    changed[0]["masks"]["tau_e"][1,0] = False
    changed[0]["targets"]["tau_e"][1,0] = 1e9
    masked = electrical_loss(model,changed,work)[0]
    changed[0]["targets"]["tau_e"][1,0] = -1e9
    assert torch.equal(masked,electrical_loss(model,changed,work)[0])
    individual = sum(electrical_loss(model,[e],work)[0] for e in data)/2
    assert torch.allclose(base,individual,atol=1e-14,rtol=1e-14)
    missing = copy.deepcopy(data); missing[0]["masks"]["tau_e"][:] = False
    try:
        electrical_loss(model,missing,work)
    except ValueError as error:
        assert "early electrical labels unavailable" in str(error)
    else:
        raise AssertionError("missing early electrical labels must reject the event")
    assert equivalence(model,data[0],work)["passed"] and equivalence(model,data[1],work)["passed"]
    assert directional(model,data[1],work)["passed"]
    batches = old.EventBatches(2)
    batches.next()
    saved = {"model":model.checkpoint(),"optimizer":old.new_optimizer(model).state_dict(),
             "numpy_batch_rng":batches.state(),"torch_rng":torch.get_rng_state()}
    a = p1a.restore(saved,2); b = p1a.restore(saved,2)
    ra = attempt(a[0],a[1],[data[1]],1701,constants["S_m"],work,0.)
    rb = p1a.attempt(b[0],b[1],[data[1]],1701,800,constants["S_m"],work)
    assert {k:v for k,v in ra.items() if k not in ("auxiliary_coefficient","auxiliary_mse")}==rb
    assert p1a.tree_equal(a[0].state_dict(),b[0].state_dict()) and p1a.tree_equal(a[1].state_dict(),b[1].state_dict())
    assert len(sequence(saved,data))==32
    print("synthetic self-check passed: t0 exclusion, truncation, original masks, event denominator, short/full states and gradients, zero-coefficient exact old update; no private assets read")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--release",type=Path)
    group.add_argument("--self-check",action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
    else:
        raise SystemExit(execute(args.release))
