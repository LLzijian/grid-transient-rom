"""Release-bound anchored-potential pair; self-check opens synthetic assets only."""
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
from kiki.models.osgkd_anchored_potential import OSGKDAnchoredPotential
from kiki.models.osgkd_fitting import IntegrationWork, integrate, event_objective, joint_loss, tensor_events

spec = importlib.util.spec_from_file_location("clean_parent_p1a", ROOT / "experiments/osgkd_v2_p1a_clean_parent_20261123/run.py")
clean = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clean)
p1a, old = clean.p1a, clean.old
require, typed_ref = clean.require, clean.typed_ref
RUN = "osgkd_v2_p1b_electrical_unlock_20261123"
PROTOCOL = "campaigns/osgkd_v2/p1b_electrical_unlock_protocol.md"
SOURCES = clean.SOURCES | {f"experiments/{RUN}/run.py", PROTOCOL,
    "src/kiki/models/osgkd_anchored_potential.py", "tests/test_osgkd_anchored_potential.py",
    "designs/osgkd_v2_anchored_electrical_potential_20261123.md"}
INPUTS = {"audit":clean.INPUTS["audit"],
    "clean_summary":"results/osgkd_v2_p1a_clean_parent_20261123/pair01/summary.json",
    "clean_analysis":"results/osgkd_v2_p1a_clean_parent_20261123/analysis.json",
    "conflict_summary":clean.INPUTS["conflict_summary"]}
CAP = 16_384_000
batch_rhs = clean.batch_rhs


def sequence(saved, fit):
    _, _, batches = clean.restore_parent(saved,len(fit))
    return [[fit[i]["id"] for i in batches.next()] for _ in range(160)]


def restore(saved, count, *, candidate=False):
    parent, optimizer, batches = clean.restore_parent(saved,count)
    if not candidate:
        return parent, optimizer, batches
    model = OSGKDAnchoredPotential.from_parent_checkpoint(saved["model"])
    initializer, dynamics, coupling = old.parameter_groups(model)
    dynamics = [p for p in dynamics if p is not model.rho]
    optimizer = torch.optim.Adam([{"params":initializer}, {"params":dynamics+coupling}])
    optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
    require(p1a.tree_equal(optimizer.state_dict(),saved["optimizer"]),"shared Adam restoration differs")
    parent_parameters, parameters = dict(parent.named_parameters()),dict(model.named_parameters())
    for name, parameter in parent_parameters.items():
        require(torch.equal(parameter,parameters[name]),"shared parameter differs: "+name)
    optimizer.param_groups[1]["params"].append(model.rho)
    require(model.rho not in optimizer.state,"new rho Adam state must be empty")
    require([g["lr"] for g in optimizer.param_groups]==[g["lr"] for g in saved["optimizer"]["param_groups"]],"parent LR differs")
    torch.set_rng_state(saved["torch_rng"])
    return model, optimizer, batches


def attempt(model, optimizer, batch, number, scale, work, multiplier=1.):
    groups = dict(zip(("initializer","dynamics","coupling"),old.parameter_groups(model),strict=True))
    if hasattr(model,"rho"):
        groups["dynamics"] = [p for p in groups["dynamics"] if p is not model.rho]
        groups["rho"] = [model.rho]
    before = {name:[p.detach().clone() for p in group] for name,group in groups.items()}
    adam, rho_gradient = {}, {}
    def gradient_hook(gradient):
        rho_gradient.update(l2=float(gradient.norm()),max_abs=float(gradient.abs().max()),finite=bool(torch.isfinite(gradient).all()))
    def post_step(opt,args,kwargs):
        adam.update({name:[p.detach().clone() for p in group] for name,group in groups.items()})
    hook = model.rho.register_hook(gradient_hook) if hasattr(model,"rho") else None
    try:
        with optimizer.register_step_post_hook(post_step):
            row = p1a.attempt(model,optimizer,batch,number,800,scale,work)
    finally:
        if hook is not None:
            hook.remove()
    metrics = None
    if row["accepted"]:
        metrics = {}
        for name,group in groups.items():
            origin = torch.cat([p.reshape(-1) for p in before[name]])
            updated = torch.cat([p.reshape(-1) for p in adam[name]])
            final = torch.cat([p.detach().reshape(-1) for p in group])
            norm = float(origin.norm())
            metrics[name] = {"parameter_l2":norm,"adam_update_l2":float((updated-origin).norm()),
                "projection_displacement_l2":float((final-updated).norm()),"final_update_l2":float((final-origin).norm()),
                "relative_final_update_l2":float((final-origin).norm())/norm if norm else None}
    norm = row["gradient_norm"]
    return {**row,"lr_multiplier":multiplier,"update_groups":metrics,"rho_gradient_before_clip":rho_gradient or None,
        "gradient_clip_multiplier":min(1.,5./(norm+1e-6)) if norm is not None and math.isfinite(norm) else None}


def validate(release):
    require((release.get("run_id"),release.get("job_id"),release.get("released"),release.get("device"),
        release.get("threads"),release.get("recompute"))==(RUN,"pair01",True,"cpu",1,False),"release contract differs")
    require(SOURCES<=release.get("source_hashes",{}).keys(),"missing source hash")
    for name,digest in release["source_hashes"].items():
        require(old._identity(old._repo_path(name))["sha256"]==digest,"source identity differs: "+name)
    require(set(release.get("input_refs",{}))==set(INPUTS),"exact input_refs required")
    inputs = {}
    for key,path in INPUTS.items():
        ref = release["input_refs"][key]
        require(ref["path"]==path,"noncanonical input: "+key)
        inputs[key] = json.loads(typed_ref(ref).read_bytes())
    summary,analysis,conflict = (inputs[k] for k in ("clean_summary","clean_analysis","conflict_summary"))
    require(summary.get("run_id")==clean.RUN and summary.get("job_id")=="pair01" and summary.get("status")=="completed"
        and summary["result"]["fixed_endpoint"]==1860 and not summary["result"]["stopped_early"],"completed Step1 epoch1860 required")
    require(analysis.get("run_id")==clean.RUN and analysis.get("job")==release["input_refs"]["clean_summary"],"analysis does not bind Step1 summary")
    choice = analysis.get("choice")
    require(choice in ("control","candidate") and analysis["gates"][choice]["passed"] is True,"Step1 must have qualified work parent")
    require(summary["release"]["input_refs"]["audit"]==release["input_refs"]["audit"],"Step1 audit differs")
    require(conflict.get("run_id")=="osgkd_v2_conflict_profile_20261123" and conflict.get("job_id")=="probe01" and conflict.get("status")=="completed"
        and conflict["release"]["input_refs"]["audit"]==release["input_refs"]["audit"],"completed shared-audit conflict diagnostic required")
    checkpoint = summary["result"]["treatments"][choice]["checkpoint"]
    require(release["checkpoint_ref"]==checkpoint==analysis["inputs"][choice]["checkpoint"],"selected Step1 checkpoint differs")
    require(checkpoint["path"]==f"artifacts/{clean.RUN}/pair01/{choice}/attempt_1860.pt","canonical selected Step1 checkpoint required")
    typed_ref(checkpoint)
    return inputs


def same_bits(a,b):
    return a.shape==b.shape and torch.equal(a.contiguous().view(torch.int64),b.contiguous().view(torch.int64))


def qualify(saved, fit, raw, constants, directory):
    order = list(dict.fromkeys(identity for batch in sequence(saved,fit) for identity in batch))
    by_id, labels = {e["id"]:e for e in fit},{e["id"]:e["label"] for e in raw}
    ordered = [by_id[identity] for identity in order]
    stable = next(e for e in ordered if labels[e["id"]]=="stable" and len(e["targets"]["theta"])==801)
    unstable = next(e for e in ordered if labels[e["id"]]=="unstable")
    parent,_,_ = restore(saved,len(fit))
    candidate,_,_ = restore(saved,len(fit),candidate=True)
    before = [copy.deepcopy(m.state_dict()) for m in (parent,candidate)]
    work = IntegrationWork()
    rows = []
    for event in (stable,unstable):
        theta,omega = (event["history"][k] for k in ("theta","omega"))
        with torch.no_grad():
            initials = [m.initialize(theta,omega) for m in (parent,candidate)]
            work.encoder_events += 2
            exact = same_bits(*initials)
            fields = [m.rhs(None,s) for m,s in zip((parent,candidate),initials,strict=True)]
            work.rhs["forward"] += 2
            exact &= same_bits(*fields)
            steps = min(800,len(event["targets"]["theta"])-1)
            for h in (.005,.0025):
                paths = [integrate(m,s,steps,internal_step=h,work=work) for m,s in zip((parent,candidate),initials,strict=True)]
                exact &= same_bits(*paths)
                for a,b in zip(parent.ports(paths[0]),candidate.ports(paths[1]),strict=True):
                    exact &= same_bits(a,b)
        losses, gradients = [],[]
        for m in (parent,candidate):
            m.zero_grad(set_to_none=True)
            components = event_objective(m,[event],800,constants["S_m"],work=work)
            losses.append(components.detach())
            joint_loss(components).backward()
            gradients.append({name:p.grad.detach().clone() for name,p in m.named_parameters() if p.grad is not None})
        exact &= same_bits(*losses)
        exact &= set(gradients[0])==set(dict(parent.named_parameters()))
        exact &= set(gradients[1])-set(gradients[0])=={"rho"}
        exact &= all(name in gradients[1] and same_bits(g,gradients[1][name]) for name,g in gradients[0].items())
        rho = gradients[1].get("rho")
        usable = rho is not None and bool(torch.isfinite(rho).all()) and float(rho.norm())>0
        rows.append({"id":event["id"],"steps":steps,"exact_embedding":bool(exact),"rho_gradient_usable":usable,
                     "rho_gradient_l2":float(rho.norm()) if rho is not None else None})
    require(all(p1a.tree_equal(state,m.state_dict()) for state,m in zip(before,(parent,candidate),strict=True)),"qualification changed model")
    numeric = p1a.numerical(parent,[stable,unstable],work)
    result = {"passed":all(r["exact_embedding"] and r["rho_gradient_usable"] for r in rows) and numeric["passed"],
        "events":rows,"parent_numerical":numeric,"optimizer_updates":0,"work":old._work_receipt(work,with_backward=True)}
    old._write_json(directory/"qualification.json",result)
    return result


def compare(saved, fit, raw, constants, directory, release):
    from kiki.evaluation.osgkd import evaluate_fit297
    order = sequence(saved,fit)
    order_ref = old._write_json(directory/"batch_sequence.json",order)
    states = {}
    for name,multiplier in (("control",1.),("candidate",1.)):
        model,opt,batches = restore(saved,len(fit), candidate=name=="candidate")
        for group in opt.param_groups:
            group["lr"] *= multiplier
        target = directory/name
        target.mkdir()
        states[name] = dict(model=model,opt=opt,batches=batches,lr_multiplier=multiplier,target=target,
                            work=IntegrationWork(),rng=torch.get_rng_state(),rows=[],consecutive=0,rejected=0,wall_s=0.,cpu_s=0.)
    stopped = False
    recovery_refs = []
    for number,expected in enumerate(order,1861):
        for state in states.values():
            torch.set_rng_state(state["rng"])
            batch = [fit[i] for i in state["batches"].next()]
            require([e["id"] for e in batch]==expected,"paired order differs")
            expected_rhs = batch_rhs(batch)
            before_rhs = state["work"].rhs["forward"]
            require(sum(s["work"].rhs["forward"] for s in states.values())+expected_rhs<=CAP, "training RHS cap would be exceeded")
            start,cpu = perf_counter(),process_time()
            row = attempt(state["model"],state["opt"],batch,number,constants["S_m"],state["work"],state["lr_multiplier"])
            require(state["work"].rhs["forward"]-before_rhs==expected_rhs,"actual training RHS differs from precheck")
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
        if number in (1892,1924,1956,1988):
            recovery = directory/"recovery"/f"attempt_{number}"
            recovery.mkdir(parents=True,exist_ok=False)
            snapshots = {}
            for name,state in states.items():
                torch.set_rng_state(state["rng"])
                snapshots[name] = old._checkpoint(recovery/f"{name}.pt",state["model"],state["opt"],
                    state["batches"],number,{"audit_ref":release["input_refs"]["audit"],"release":release,
                                           "recovery_only":True},torch.device("cpu"))
            recovery_refs.append(old._write_json(recovery/"manifest.json",{
                "run_id":RUN,"job_id":"pair01","attempt":number,"recovery_only":True,
                "completed_pairs":number-1860,"checkpoints":snapshots,
                "completed_utc":datetime.now(timezone.utc).isoformat()}))
    outputs = {}
    for name,state in states.items():
        torch.set_rng_state(state["rng"])
        checkpoint = old._checkpoint(state["target"]/f"attempt_{number}.pt",state["model"],state["opt"],state["batches"],number,
                                     {"audit_ref":release["input_refs"]["audit"],"release":release},torch.device("cpu"))
        outputs[name] = {"checkpoint":checkpoint,"attempts":len(state["rows"]),"accepted":len(state["rows"])-state["rejected"],
                         "rejected":state["rejected"],"clipped":sum(r["clipped"] for r in state["rows"]),
                         "work":old._work_receipt(state["work"],with_backward=True),
                         "attempts_ref":old._write_json(state["target"]/"attempts.json",state["rows"]),
                         "train_cost":{"wall_s":state["wall_s"],"cpu_s":state["cpu_s"]}}
    rhs = sum(s["work"].rhs["forward"] for s in states.values())
    require(rhs<=CAP,"paired training RHS cap exceeded")
    if not stopped:
        for name,state in states.items():
            target = state["target"]/"evaluation"; target.mkdir()
            with torch.no_grad():
                scores = evaluate_fit297(state["model"].eval(),raw,constants,output_directory=target)
            outputs[name].update(evaluation_cost=scores["cost"],evaluation_summary_ref=old._identity(target/"summary.json"))
    return {"treatments":outputs,"target_endpoint":2020,"actual_endpoint":number,"fixed_endpoint":2020 if not stopped else None,
            "stopped_early":stopped,"paired_batch_sequence_ref":order_ref,"recovery_manifest_refs":recovery_refs,
            "training_rhs_forward":rhs,"training_rhs_cap":CAP,"new_fom":0}



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
        raw,constants = p1a.load_fit297(release["input_refs"]["audit"])
        require(len(raw)==297 and sum(e["label"]=="unstable" for e in raw)==30,"fit297 denominator differs")
        saved = torch.load(typed_ref(release["checkpoint_ref"]),map_location="cpu",weights_only=True)
        require(saved["attempt"]==1860 and saved["model"]["config"]["variant"]=="C1" and
                saved["provenance"]["audit_ref"]==release["input_refs"]["audit"],"audited selected C1 epoch1860 required")
        sides,excluded = p1a.fitting_events(raw,constants)
        qualification = qualify(saved,sides["fit"],raw,constants,local)
        require(qualification["passed"],"zero embedding real qualification failed; no training permitted")
        result = compare(saved,sides["fit"],raw,constants,local,release)
        result.update(qualification_ref=old._identity(local/"qualification.json"),
                      objective_excluded_ids=excluded,evaluation_denominator=297)
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
    import tempfile
    from unittest.mock import patch
    torch.set_num_threads(1)
    constants = {key:np.ones(10) for key in ("mass","damping","Sp","Se","omega_scale","S_m")}
    constants.update(F=np.eye(10),theta_ref=np.zeros(10),edge_index=np.array([[0],[1]]))
    model = old.new_model("C1",constants,torch.device("cpu"))
    def event(identity,count,label):
        theta = np.tile(np.linspace(.4,-.3,10),(11,1)) + np.arange(11)[:,None]*.001
        omega = np.tile(np.linspace(.1,-.05,10),(11,1))
        return {"id":identity,"label":label,"history":{"theta":theta,"omega":omega},
            "t":np.arange(count+1)*.01,"targets":{"theta":np.tile(theta[-1],(count+1,1)),
                "omega":np.tile(omega[-1],(count+1,1)),"tau_m":np.zeros((count+1,10)),"tau_e":np.zeros((count+1,10))},
            "masks":{k:np.ones((count+1,10),bool) for k in ("theta","omega","tau_m","tau_e")}}
    short = tensor_events([event("warm",3,"unstable")],"cpu",max_steps=800)
    optimizer = old.new_optimizer(model)
    assert p1a.attempt(model,optimizer,short,1860,800,constants["S_m"],IntegrationWork())["accepted"]
    batches = old.EventBatches(2); batches.next()
    saved = {"model":model.checkpoint(),"optimizer":copy.deepcopy(optimizer.state_dict()),
        "numpy_batch_rng":batches.state(),"torch_rng":torch.get_rng_state().clone()}
    original = copy.deepcopy(saved)
    parent,parent_opt,_ = restore(saved,2)
    measured = attempt(parent,parent_opt,short,1861,constants["S_m"],IntegrationWork())
    plain,plain_opt,_ = restore(saved,2)
    ordinary = p1a.attempt(plain,plain_opt,short,1861,800,constants["S_m"],IntegrationWork())
    assert all(measured[k]==v for k,v in ordinary.items())
    assert p1a.tree_equal(parent.state_dict(),plain.state_dict())
    assert p1a.tree_equal(parent_opt.state_dict(),plain_opt.state_dict())
    candidate,candidate_opt,_ = restore(saved,2,candidate=True)
    assert candidate.rho not in candidate_opt.state
    row = attempt(candidate,candidate_opt,short,1861,constants["S_m"],IntegrationWork())
    assert row["accepted"] and row["rho_gradient_before_clip"]["l2"]>0
    assert row["update_groups"]["rho"]["final_update_l2"]>0
    assert candidate.rho in candidate_opt.state
    assert set(row["update_groups"])=={"initializer","dynamics","coupling","rho"}
    raw = [event("long",800,"stable"),event("short",3,"unstable")]
    fit = tensor_events(raw,"cpu",max_steps=800)
    assert len(sequence(saved,fit))==160
    with tempfile.TemporaryDirectory(prefix="synthetic_",dir=Path(__file__).parent) as directory:
        qualification = qualify(saved,fit,raw,constants,Path(directory))
        refs = {key:{"path":path,"bytes":0,"sha256":"a"*64} for key,path in INPUTS.items()}
        checkpoint = {"path":f"artifacts/{clean.RUN}/pair01/control/attempt_1860.pt","bytes":0,"sha256":"a"*64}
        release = {"run_id":RUN,"job_id":"pair01","released":True,"device":"cpu","threads":1,"recompute":False,
                   "source_hashes":dict.fromkeys(SOURCES,"a"*64),"input_refs":refs,"checkpoint_ref":checkpoint}
        fixtures = {"audit":{},"clean_summary":{"run_id":clean.RUN,"job_id":"pair01","status":"completed",
            "result":{"fixed_endpoint":1860,"stopped_early":False,"treatments":{"control":{"checkpoint":checkpoint}}},
            "release":{"input_refs":{"audit":refs["audit"]}}},
            "clean_analysis":{"run_id":clean.RUN,"job":refs["clean_summary"],"choice":"control",
                "gates":{"control":{"passed":True}},"inputs":{"control":{"checkpoint":checkpoint}}},
            "conflict_summary":{"run_id":"osgkd_v2_conflict_profile_20261123","job_id":"probe01","status":"completed",
                "release":{"input_refs":{"audit":refs["audit"]}}}}
        files = {}
        for key,payload in fixtures.items():
            target = Path(directory)/f"{key}.json"
            target.write_text(json.dumps(payload),encoding="utf-8")
            files[refs[key]["path"]] = target
        files[checkpoint["path"]] = Path(directory)/"unused.pt"
        with patch.dict(globals(),{"typed_ref":lambda ref:files[ref["path"]]}), \
             patch.object(old,"_identity",return_value={"sha256":"a"*64}):
            assert validate(release)["clean_analysis"]["choice"]=="control"
            fixtures["clean_analysis"]["choice"] = "parent"
            files[refs["clean_analysis"]["path"]].write_text(json.dumps(fixtures["clean_analysis"]),encoding="utf-8")
            try:
                validate(release)
            except ValueError:
                pass
            else:
                raise AssertionError("unqualified Step1 choice accepted")
    assert qualification["passed"] and qualification["optimizer_updates"]==0
    assert p1a.tree_equal(saved,original)
    try:
        validate({})
    except ValueError:
        pass
    else:
        raise AssertionError("unreleased execution accepted")
    print("synthetic self-check passed: shared nonempty Adam exact restoration, empty rho state and inherited LR, measured/plain control equality, rho gradient/update, fixed160 batches, full8s/short no-update zero qualification; no private assets opened")


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
