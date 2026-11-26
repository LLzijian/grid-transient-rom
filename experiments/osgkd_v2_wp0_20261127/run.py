"""Released zero-update WP0 assessment of one val event and train edge rankings."""

import argparse
from pathlib import Path
import sys
from time import perf_counter, process_time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/"src")]

import torch

from experiments.osgkd_v2_sprint2a_20261125 import resume_host as io, run as matched
from experiments.osgkd_v2_sprint2a_20261125.score import OSGKDGraph, OSGKDMatchedFlat
from experiments.osgkd_v2_wp0_20261127 import edge_audit, pilot

RUN = "osgkd_v2_wp0_20261127"
PROTOCOL = "campaigns/osgkd_v2/wp0_mechanism_rank_protocol.md"
PARENT = "results/osgkd_v2_sprint2a_20261125/evaluate01/summary.json"
PARENT_SHA = "7645c8b006b9dd6e71838e7e875da69117ed3df9011938d36f46cf126c6b3fbb"
TRACE_AUTHORITY = "results/osgkd_v2_p2a_post_audit_20261124/probe01/summary.json"
TRACE_AUTHORITY_SHA = "5381143c686e4739b749ffcb76a48d322f4332fa3720f5a8425160346c62e6a1"


def validate(path):
    release = io.read(path)
    io.require(path == ROOT/"results"/RUN/"release.json" and
               (release["run_id"],release["released"],release["official_updates"],release["new_fom"])==
               (RUN,True,0,0), "WP0 released authority required")
    for source,digest in release["source_hashes"].items():
        document = (ROOT/source).resolve()
        io.require(document.is_relative_to(ROOT) and io.identity(document)["sha256"]==digest,
                   "bound source differs: "+source)
    for key,location,digest in (("parent",PARENT,PARENT_SHA),
                                 ("historical_pilot",TRACE_AUTHORITY,TRACE_AUTHORITY_SHA)):
        ref = release[key]
        io.require(ref["path"]==location and ref["sha256"]==digest, "fixed input differs")
        io.verified(ref)
    io.require(release["scientist"]["configured_model"]=="gpt-6-astra" and
               release["independent_preflight"]["status"]=="passed", "recorded separate scientific review required")
    return release


def restore(parent):
    models = {}
    for name,(kind,_) in matched.JOBS.items():
        receipt = io.read(io.verified(parent["release"]["fit_refs"][name]))
        terminal = receipt["result"]
        ref = terminal["checkpoints"]["2048"]
        io.require(parent["result"]["val"][name]["checkpoint_ref"]==ref and
                   terminal["valid_endpoint"] and terminal["accepted"]==2048,"parent endpoint is not fixed")
        model_type = OSGKDGraph if kind=="graph" else OSGKDMatchedFlat
        saved = torch.load(io.verified(ref),map_location="cpu",weights_only=True)
        io.require(saved["attempt"]==2048 and saved["release"]==receipt["release"], "parent checkpoint authority differs")
        models[name] = model_type.from_checkpoint(saved["model"]).eval().requires_grad_(False)
    return models


def public_pilot(audit, private_ref):
    clean = {key:value for key,value in audit.items() if key != "models"}
    clean["details_ref"] = private_ref
    clean["models"] = {}
    for name,record in audit["models"].items():
        row = {key:value for key,value in record.items() if key not in
               ("latent_t0","latent_t0_normalized","force","scale_only")}
        row["latent_t0_normalized_abs_max"] = [max(map(abs,block)) for block in record["latent_t0_normalized"]]
        row["force"] = {key:{stat:value for stat,value in block.items()
                             if stat not in ("node_t0","node_impulse")}
                        for key,block in record["force"].items()}
        alternative = record["scale_only"]
        row["scale_only"] = {key:value for key,value in alternative.items() if key != "latent_t0"} if alternative else None
        clean["models"][name] = row
    return clean


def public_edges(audit, private_ref):
    summary = {}
    for name,row in audit["per_seed"].items():
        detail = io.read(io.verified(row["details_ref"]))
        summary[name] = dict(details_ref=row["details_ref"],work_ref=row["work_ref"],
                             diagnostic_rhs=row["diagnostic_rhs"],
                             static_terms={key:detail["groups"][key] for key in ("environment_a","environment_a2")})
    return dict(statistics=audit["statistics"],per_seed=summary,details_ref=private_ref)


def execute(path):
    release = validate(path)
    local = ROOT/"artifacts"/RUN/"probe01"
    public = ROOT/"results"/RUN/"probe01"
    io.require(not local.exists() and not public.exists(), "WP0 execution already claimed")
    local.mkdir(parents=True)
    public.mkdir(parents=True)
    ref = io.identity(path)
    io.write(local/"started.json",dict(release=ref,started_utc=io.utc()))
    started,cpu = perf_counter(),process_time()
    phase = "train_identity"
    try:
        torch.set_num_threads(1)
        parent = io.read(io.verified(release["parent"]))
        models = restore(parent)
        anchor = io.read(io.verified(parent["release"]["input_refs"]["anchor_analysis"]))
        train,constants = matched.load_train({"anchor_analysis":anchor})
        recent = io.read(io.verified(release["historical_pilot"]))
        trace = recent["result"]["fixed_val_false_alarm"]["trace_ref"]
        phase = "single_pilot_prefix"
        result_pilot = pilot.audit(models,parent["result"]["val"],constants,trace,local)
        io.require(result_pilot["work"]["rhs_forward"]==13*24*pilot.STEPS,"planned prefix work differs")
        io.write(local/"pilot_diagnostics.json",result_pilot)
        phase = "three_seed_train_contributions"
        graph = {key:value for key,value in models.items() if key.startswith("graph_")}
        result_edges = edge_audit.audit(graph,train,local/"edge_contributions")
        io.write(local/"edge_diagnostics.json",result_edges)
        summary = {"run_id":RUN,"status":"completed","job_id":"probe01","release":ref,
                   "official_updates":0,"new_fom":0,"test_trajectory_access":False,
                   "val_scope":"one already exposed pilot, 11 motion frames; earlier frozen truth",
                   "pilot":public_pilot(result_pilot,io.identity(local/"pilot_diagnostics.json")),
                   "edges":public_edges(result_edges,io.identity(local/"edge_diagnostics.json")),
                   "completed_utc":io.utc(),
                   "wall_s":perf_counter()-started,"cpu_s":process_time()-cpu}
        summary_ref = matched.baseline.atomic_write(public/"summary.json",summary,public=True)
        io.write(local/"completed.json",dict(summary=summary_ref,status="completed"))
        print(summary_ref,flush=True)
    except BaseException as error:
        io.write(local/"failed.json",dict(status="failed",phase=phase,error=repr(error),release=ref))
        raise


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release",required=True,type=Path)
    execute(parser.parse_args().release.resolve())
