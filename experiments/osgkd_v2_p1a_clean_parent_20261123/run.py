"""Release-bound clean-parent LR pair; synthetic checks never open fit assets."""
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
RUN = "osgkd_v2_p1a_clean_parent_20261123"
SOURCES = (p1a.SOURCES - {p1a.PROTOCOL}) | {f"experiments/{RUN}/run.py",
    "experiments/osgkd_v2_p1a_20261123/run.py"}
INPUTS = {"audit":"results/osgkd_fit297_20261113/port_audit_release.json",
          "early_summary":"results/osgkd_v2_early_electrical_20261123/pair01/summary.json",
          "conflict_summary":"results/osgkd_v2_conflict_profile_20261123/probe01/summary.json"}
CAP = 13_107_200


def require(value, message):
    if not value:
        raise ValueError(message)



def sequence(saved, fit):
    _, _, batches = p1a.restore(saved,len(fit))
    return [[fit[i]["id"] for i in batches.next()] for _ in range(128)]


def restore_parent(saved, count):
    model,opt,batches = p1a.restore(saved,count)
    require(p1a.tree_equal(model.checkpoint(),saved["model"]),"restored model differs")
    require(p1a.tree_equal(opt.state_dict(),saved["optimizer"]),"restored Adam differs")
    require(p1a.tree_equal(batches.state(),saved["numpy_batch_rng"]),"restored batch RNG differs")
    require(torch.equal(torch.get_rng_state(),saved["torch_rng"]),"restored Torch RNG differs")
    return model,opt,batches


def batch_rhs(batch):
    return 8*sum(min(800,len(e["targets"]["theta"])-1) for e in batch)


def attempt(model, optimizer, batch, number, scale, work, multiplier):
    # Hooks copy values only: the frozen optimizer and projection retain their exact order.
    groups = dict(zip(("initializer","dynamics","coupling"),old.parameter_groups(model),strict=True))
    before = {name:[p.detach().clone() for p in group] for name,group in groups.items()}
    adam = {}
    def post_step(opt, args, kwargs):
        adam.update({name:[p.detach().clone() for p in group] for name,group in groups.items()})
    with optimizer.register_step_post_hook(post_step):
        row = p1a.attempt(model,optimizer,batch,number,800,scale,work)
    metrics = None
    if row["accepted"]:
        metrics = {}
        for name,group in groups.items():
            origin = torch.cat([p.reshape(-1) for p in before[name]])
            pre_projection = torch.cat([p.reshape(-1) for p in adam[name]])
            final = torch.cat([p.detach().reshape(-1) for p in group])
            norm = float(origin.norm())
            update = float((final-origin).norm())
            metrics[name] = {"parameter_l2":norm,"adam_update_l2":float((pre_projection-origin).norm()),
                "projection_displacement_l2":float((final-pre_projection).norm()),"final_update_l2":update,
                "relative_adam_update_l2":float((pre_projection-origin).norm())/norm if norm else None,
                "relative_projection_displacement_l2":float((final-pre_projection).norm())/norm if norm else None,
                "relative_final_update_l2":update/norm if norm else None}
    norm = row["gradient_norm"]
    return {**row,"lr_multiplier":multiplier,"update_groups":metrics,
        "gradient_clip_multiplier":min(1.,5./(norm+1e-6)) if norm is not None and math.isfinite(norm) else None}


def typed_ref(ref):
    require(isinstance(ref,dict) and set(ref)=={"path","bytes","sha256"},"exact typed ref required")
    require(type(ref["bytes"]) is int and ref["bytes"]>=0 and isinstance(ref["sha256"],str)
        and len(ref["sha256"])==64 and all(c in "0123456789abcdef" for c in ref["sha256"]),"invalid ref types")
    return old._verified_ref(ref)


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
    early,conflict = inputs["early_summary"],inputs["conflict_summary"]
    require(early.get("run_id")=="osgkd_v2_early_electrical_20261123" and early.get("job_id")=="pair01"
        and early.get("status")=="completed" and early["result"]["fixed_endpoint"]==1732
        and not early["result"]["stopped_early"],"completed official electrical pair1732 required")
    require(conflict.get("run_id")=="osgkd_v2_conflict_profile_20261123" and conflict.get("job_id")=="probe01"
        and conflict.get("status")=="completed","completed conflict diagnostic required")
    require(early["release"]["audit_ref"]==release["input_refs"]["audit"],"parent audit differs")
    require(conflict["release"]["input_refs"]["audit"]==release["input_refs"]["audit"],"diagnostic audit differs")
    require(release["checkpoint_ref"]==early["result"]["treatments"]["control"]["checkpoint"],"clean control1732 required")
    require(release["checkpoint_ref"]["path"]=="artifacts/osgkd_v2_early_electrical_20261123/pair01/control/attempt_1732.pt",
        "canonical clean control checkpoint required")
    typed_ref(release["checkpoint_ref"])


def self_check():
    torch.set_num_threads(1)
    constants = {key:np.ones(2) for key in ("mass","damping","Sp","Se","omega_scale","S_m")}
    constants.update(F=np.eye(2),theta_ref=np.zeros(2),edge_index=np.array([[0],[1]]))
    model = old.new_model("C1",constants,torch.device("cpu"))
    count = 3
    event = {"id":"synthetic","history":{k:np.zeros((11,2)) for k in ("theta","omega")},
        "t":np.arange(count+1)*.01,"targets":{k:np.zeros((count+1,2)) for k in ("theta","omega","tau_m","tau_e")},
        "masks":{k:np.ones((count+1,2),bool) for k in ("theta","omega","tau_m","tau_e")}}
    data = tensor_events([event],"cpu",max_steps=800)
    optimizer = old.new_optimizer(model)
    p1a.attempt(model,optimizer,data,1732,800,constants["S_m"],IntegrationWork())
    batches = old.EventBatches(1); batches.next()
    saved = {"model":model.checkpoint(),"optimizer":copy.deepcopy(optimizer.state_dict()),
        "numpy_batch_rng":batches.state(),"torch_rng":torch.get_rng_state().clone()}
    a = restore_parent(saved,1)
    work = IntegrationWork()
    row = attempt(a[0],a[1],data,1733,constants["S_m"],work,1.)
    rng = torch.get_rng_state().clone()
    b = p1a.restore(saved,1)
    plain = p1a.attempt(b[0],b[1],data,1733,800,constants["S_m"],IntegrationWork())
    assert all(row[k]==v for k,v in plain.items()) and row["accepted"]
    assert p1a.tree_equal(a[0].state_dict(),b[0].state_dict())
    assert p1a.tree_equal(a[1].state_dict(),b[1].state_dict()) and torch.equal(rng,torch.get_rng_state())
    assert work.rhs["forward"]==batch_rhs(data)==24
    assert a[2].state()==b[2].state() and len(sequence(saved,data))==128
    assert set(row["update_groups"])=={"initializer","dynamics","coupling"}
    print("synthetic self-check passed: nonempty Adam restoration, exact measured/plain update, optimizer and Torch RNG equality, batch restoration and actual RHS; no private assets read")


def compare(saved, fit, raw, constants, directory, release):
    from kiki.evaluation.osgkd import evaluate_fit297
    order = sequence(saved,fit)
    order_ref = old._write_json(directory/"batch_sequence.json",order)
    states = {}
    for name,multiplier in (("control",1.),("candidate",.3)):
        model,opt,batches = restore_parent(saved,len(fit))
        for group in opt.param_groups:
            group["lr"] *= multiplier
        target = directory/name
        target.mkdir()
        states[name] = dict(model=model,opt=opt,batches=batches,lr_multiplier=multiplier,target=target,
                            work=IntegrationWork(),rng=torch.get_rng_state(),rows=[],consecutive=0,rejected=0,wall_s=0.,cpu_s=0.)
    stopped = False
    recovery_refs = []
    for number,expected in enumerate(order,1733):
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
        if number in (1764,1796,1828):
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
                "completed_pairs":number-1732,"checkpoints":snapshots,
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
    return {"treatments":outputs,"target_endpoint":1860,"actual_endpoint":number,"fixed_endpoint":1860 if not stopped else None,
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
        saved = torch.load(old._verified_ref(release["checkpoint_ref"]),map_location="cpu",weights_only=True)
        require(saved["attempt"]==1732 and saved["model"]["config"]["variant"]=="C1" and
                saved["provenance"]["audit_ref"]==release["input_refs"]["audit"],"audited clean-control C1 attempt1732 required")
        sides,excluded = p1a.fitting_events(raw,constants)
        require([g["lr"] for g in saved["optimizer"]["param_groups"]]==[1e-3,3e-4],"parent learning rates differ")
        result = compare(saved,sides["fit"],raw,constants,local,release)
        result.update(objective_excluded_ids=excluded,evaluation_denominator=297)
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
