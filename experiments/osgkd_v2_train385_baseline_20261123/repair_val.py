"""One release-bound val torque-coordinate repair, preserving the completed train result."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter, process_time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import h5py
import numpy as np
import torch
from experiments.osgkd_v2_train385_baseline_20261123 import run as baseline
from kiki.data import osgkd_train_val as packets
from kiki.models.osgkd import OSGKD

RUN = baseline.RUN
PROTOCOL = "campaigns/osgkd_v2/train385_baseline_val_alignment.md"
SOURCES = baseline.SOURCES | {f"experiments/{RUN}/repair_val.py", PROTOCOL}
ORIGINAL_SHA = "099a3b168a390d0d183dc84e19bdd08e4e6731d248a966e20dabac4a7aeea542"
COORDINATE = "tau_m=Pm-Pm_ref;tau_e=te-Pm_ref"
require, read, verified = baseline.require, baseline.read, baseline.verified


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()


def without_ports(value, *, drop_cost=False):
    if isinstance(value, dict):
        return {key:without_ports(item, drop_cost=drop_cost) for key,item in value.items()
                if key not in ("tau_m", "tau_e") and not (drop_cost and key == "cost")}
    if isinstance(value, list):
        return [without_ports(item,drop_cost=drop_cost) for item in value]
    return value


def assert_equivalent(before_rows, after_rows, before, after):
    require(canonical(without_ports(before_rows)) == canonical(without_ports(after_rows)),
            "per-event motion or classification changed")
    require(canonical(without_ports(before["summary"],drop_cost=True)) ==
            canonical(without_ports(after["summary"],drop_cost=True)), "aggregate motion or classification changed")
    require(canonical(before["timing"]) == canonical(after["timing"]), "timing changed")


def validate(release):
    require(tuple(release.get(k) for k in ("run_id","job_id","released","device","threads","recompute")) ==
            (RUN,"eval02",True,"cpu",1,False), "fixed CPU eval02 release required")
    require(SOURCES <= release.get("source_hashes",{}).keys(), "source hashes missing")
    for name, sha in release["source_hashes"].items():
        require(baseline.identity(baseline.repo_path(name))["sha256"] == sha, "source changed: " + name)
    ref = release["original_summary_ref"]
    require(ref["path"] == f"results/{RUN}/eval01/summary.json" and ref["sha256"] == ORIGINAL_SHA,
            "exact original eval01 required")
    original = read(ref)
    require((original["run_id"],original["job_id"],original["status"]) == (RUN,"eval01","completed"),
            "completed eval01 required")
    require(release["checkpoint_ref"] == original["release"]["checkpoint_ref"] and
            release["checkpoint_ref"]["sha256"] == baseline.CHECKPOINT_SHA, "checkpoint changed")
    verified(release["checkpoint_ref"])
    for split in ("train","val"):
        for key in ("rows_ref","metrics_ref","completed_ref"):
            verified(original["result"][split][key])
    return original


def frozen_constants(original):
    parent = read(original["release"]["input_refs"]["clean_summary"])
    audit_ref = parent["release"]["input_refs"]["audit"]
    require(audit_ref["sha256"] == packets.RELEASE_SHA256, "frozen audit release changed")
    audit = read(audit_ref)
    original_audit = read(audit["original_audit"])
    constants = original_audit["constants"]
    require(constants["constants_id"] == audit["constants_id"], "frozen constants differ")
    return constants


def original_packets(events):
    """Reconstruct old port coordinates from raw val only, avoiding lossy inverse subtraction."""
    old = copy.deepcopy(events)
    by_source = {}
    for event in old:
        if event["history"] is not None:
            by_source.setdefault(event["metadata"]["source_root"], []).append(event)
    for source, selected in by_source.items():
        with h5py.File(ROOT / "private_data" / source / "dataset.h5", "r") as file:
            offset = file["equilibrium"]["Pm"][...]
            require(hashlib.sha256(np.asarray(offset,dtype="<f8").tobytes(order="C")).hexdigest() ==
                    packets.PM_REFERENCE_SHA256, "val/train common torque offset differs")
            for event in selected:
                for key, raw_key in (("tau_m","Pm"),("tau_e","te")):
                    raw = file["events"][event["id"]][raw_key][...][10:]
                    corrected = next(e for e in events if e["id"] == event["id"])
                    require(np.array_equal(raw-offset,corrected["targets"][key]), "port centering differs")
                    event["targets"][key] = raw
    return old


def execute(path):
    start,cpu = perf_counter(),process_time()
    path = path.resolve(); require(path.is_relative_to(ROOT), "repository release required")
    require(path == ROOT / "results" / RUN / "eval02_release.json", "canonical eval02 release required")
    release = json.loads(path.read_bytes()); original = validate(release)
    local,public = (ROOT / side / RUN / "eval02" for side in ("artifacts","results"))
    require(not local.exists() and not public.exists(), "eval02 already claimed; no rerun")
    local.mkdir(parents=True,exist_ok=False); public.mkdir(parents=True,exist_ok=False)
    receipt = {"run_id":RUN,"job_id":"eval02","status":"started","release":release,
               "release_ref":baseline.identity(path),"pid":os.getpid(),
               "started_utc":datetime.now(timezone.utc).isoformat(),
               "command":[sys.executable,*sys.argv],"environment":{"python":sys.version.split()[0],
               "torch":str(torch.__version__),"numpy":np.__version__,"device":"cpu","dtype":"float64","threads":1}}
    baseline.atomic_write(local / "started.json",receipt)
    try:
        torch.set_num_threads(1)
        constants = frozen_constants(original)
        events = packets.load_val(constants)
        old_events = original_packets(events)
        require(baseline.data_identity(old_events) == original["result"]["val"]["data_identity"],
                "original input histories, targets, masks, or identities changed")
        del old_events
        saved = torch.load(verified(release["checkpoint_ref"]),map_location="cpu",weights_only=True)
        require(saved["attempt"] == 1860 and saved["model"]["config"]["variant"] == "C1", "parent model differs")
        model = OSGKD.from_checkpoint(saved["model"]).cpu().double().eval().requires_grad_(False)
        for key in ("mass","damping","F","Sp","Se","omega_scale","theta_ref"):
            expected = np.asarray(constants[key])
            if key in ("mass","damping"):
                expected = np.diag(expected)
            require(np.array_equal(getattr(model,key).numpy(),expected), "model constant changed: " + key)
        require(np.array_equal(model.edge_index.numpy(),np.stack(np.triu_indices(10,k=1))), "K10 graph changed")
        planned_rhs = sum(24*(max(800,min(1500,len(e["t"])-1)) if e["label"] == "unstable" else 800)
                          for e in events if e["history"] is not None)
        require(planned_rhs == 2_054_400 == original["result"]["val"]["summary"]["work"]["rhs_forward"],
                "val horizon or RHS budget changed")
        corrected = baseline.score_split(model,events,constants,"val",local / "val")
        before_rows = read(original["result"]["val"]["rows_ref"])
        after_rows = read(corrected["rows_ref"])
        assert_equivalent(before_rows,after_rows,original["result"]["val"],corrected)
        require(corrected["data_identity"] != original["result"]["val"]["data_identity"], "port coordinate digest unchanged")
        work = corrected["summary"]["work"]
        require(work == original["result"]["val"]["summary"]["work"] and work["encoder_events"] == 107,
                "val work differs")
        receipt.update(status="completed",result={"train":original["result"]["train"],"val":corrected},
            correction={"original_summary_ref":release["original_summary_ref"],"motion_label_equivalence":True,
                        "coordinate":COORDINATE}, original_job_cost=original["cost"],
            phases={"training":{"updates":0,"event_paths":0,"rhs_forward":0},
                    "evaluation":{"updates":0,"event_paths":214,"rhs_forward":work["rhs_forward"]},"new_fom":0},
            cost={"wall_s":perf_counter()-start,"cpu_s":process_time()-cpu},
            completed_utc=datetime.now(timezone.utc).isoformat())
        ref = baseline.atomic_write(public / "summary.json",receipt,public=True)
        baseline.atomic_write(local / "completed.json",{"status":"completed","summary_ref":ref})
    except Exception as error:
        baseline.atomic_write(local / "failed.json",{"status":"failed","error":repr(error)})
        raise


def self_check():
    from kiki.data.osgkd import event_record
    constants = dict(mass=np.ones(10))
    t=np.arange(21)*.01; theta=np.zeros((21,10)); omega=theta.copy()
    offset=np.arange(10,dtype=np.float64); raw=np.ones((21,10))*3+offset
    centered = event_record("synthetic","synthetic","stable",t,theta,omega,raw-offset,
                            raw-offset,raw-offset,{},constants["mass"])
    original = event_record("synthetic","synthetic","stable",t,theta,omega,raw,raw,raw,{},constants["mass"])
    for key in ("theta","omega"):
        assert np.array_equal(centered["history"][key],original["history"][key])
        assert np.array_equal(centered["targets"][key],original["targets"][key])
    for key in ("tau_m","tau_e"):
        assert np.array_equal(centered["targets"][key],np.full((11,10),3.))
    before = {"summary":{"motion":1.,"tau_m":{"rmse":4.},"cost":{"wall_s":1}},"timing":{"hits":1}}
    after = {"summary":{"motion":1.,"tau_m":{"rmse":2.},"cost":{"wall_s":2}},"timing":{"hits":1}}
    assert_equivalent([{"motion":1.,"tau_e":4.}],[{"motion":1.,"tau_e":2.}],before,after)
    after["summary"]["motion"] = 2.
    try:
        assert_equivalent([],[],before,after)
    except ValueError:
        pass
    else:
        raise AssertionError("motion change accepted")
    print(json.dumps({"self_check":"passed","scientific_payloads_opened":False}))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("self-check"); sub.add_parser("evaluate").add_argument("--release",required=True,type=Path)
    args=parser.parse_args()
    self_check() if args.command == "self-check" else execute(args.release)
