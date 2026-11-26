"""One original-val pass for a released refit wave and frozen parents."""

import argparse
import math
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT),str(ROOT/"src")]

import torch

from experiments.osgkd_v2_p2a_equivariant_graph_20261124 import run as graph_run
from experiments.osgkd_v2_refits import fit_run
from experiments.osgkd_v2_sprint2a_20261125 import resume_host as io, run as matched
from experiments.osgkd_v2_sprint2a_20261125.score import endpoint_spectrum
from experiments.osgkd_v2_sprint2b_baselines_20261126 import score as measured
from experiments.osgkd_v2_wp0_20261127.run import restore
from kiki.data.osgkd_train_val import load_val
from kiki.models.osgkd_restricted import RestrictedOSGKDGraph

RUN = "osgkd_v2_wp1a_refits_20261127"


def aggregate(scored):
    scopes = scored["summary"]["splits"]["val"]
    rows = {}
    for name,value in scopes.items():
        if not (name.startswith("stable_") or name == "unstable_actual"):
            continue
        metric = value["official"] if name.startswith("stable_") else value["covered_only_diagnostic"]
        rows[name] = {
            "risk": metric["motion_risk_v1"] if metric else None,
            "common_speed_rmse_rad_s": metric["common_speed_rmse_rad_s"] if metric else None,
            "normalized_tau_m_rmse": metric["tau_m"]["normalized_rmse"] if metric else None,
            "normalized_tau_e_rmse": metric["tau_e"]["normalized_rmse"] if metric else None,
            "covered": value["covered"], "denominator": value["denominator"],
        }
    auto = scored["autonomous"]
    stable,unstable = auto["stable_false_alarm_8s"],auto["unstable_recall_8s"]
    io.require((stable["raw_denominator"],unstable["input_eligible_denominator"],unstable["raw_denominator"])==
               (95,12,22),"original val denominators differ")
    return dict(scopes=rows,autonomous=auto,qualified=bool(scored["gate_passed"]),
                qualified_events=scored["numerical_qualified_events"],rows_ref=scored["rows_ref"],
                metrics_ref=scored["metrics_ref"],completed_ref=scored["completed_ref"])


def refit_model(terminal, authority):
    f=terminal["result"]
    ref=f["checkpoints"]["128"]
    saved=torch.load(io.verified(ref),map_location="cpu",weights_only=True)
    io.require(saved["attempt"]==128 and saved["release_ref"]==authority
               and saved["job"]==terminal["job"] and saved["parent_ref"]==f["parent_ref"]
               and saved["plan_ref"]==f["plan_ref"],
               "refit endpoint authority differs")
    return RestrictedOSGKDGraph.from_checkpoint(saved["model"]).eval().requires_grad_(False)


def full_guard(parent,full,limits):
    if not full or not full.get("qualified"):
        return {"passed":False,"reason":"full_not_numerically_qualified"}
    a,b=parent["scopes"]["stable_8s"]["risk"],full["scopes"]["stable_8s"]["risk"]
    before,after=parent["autonomous"],full["autonomous"]
    fp_delta=(after["stable_false_alarm_8s"]["autonomous_trigger"]-
              before["stable_false_alarm_8s"]["autonomous_trigger"])
    recall_drop=(before["unstable_recall_8s"]["autonomous_trigger"]-
                 after["unstable_recall_8s"]["autonomous_trigger"])
    growth_change=full["spectrum"].get("max_real")
    source_alpha=parent["spectrum"].get("max_real")
    increase=(growth_change-source_alpha) if growth_change is not None and source_alpha is not None else None
    gates=dict(
        stable_risk=(a is not None and b is not None and
                     b <= a+max(limits["stable_8s_abs"],limits["stable_8s_rel"]*a)),
        stable_false_alarms=fp_delta<=limits["additional_stable_fp"],
        recall=recall_drop<=limits["unstable_hit_drop"],
        spectrum=(bool(full["spectrum"]["qualified"]) and increase is not None
                  and increase<=limits["positive_real_growth_increase"]),
    )
    return dict(passed=all(gates.values()),gates=gates,stable_risk_before=a,stable_risk_after=b,
                false_alarm_increase=fp_delta,autonomous_recall_drop=recall_drop,max_real_increase=increase)


def hypergeom_agreement(a,b,n=45):
    if not a and not b:
        return dict(intersection=0,f1=None,jaccard=None,expected_f1=None,expected_jaccard=None)
    k,l=len(a),len(b)
    lo,hi=max(0,k+l-n),min(k,l)
    denom=math.comb(n,l)
    expected_j=sum((i/(k+l-i))*math.comb(k,i)*math.comb(n-k,l-i)/denom
                   for i in range(lo,hi+1))
    inter=len(a&b)
    return dict(intersection=inter,first_edges=k,second_edges=l,f1=2*inter/(k+l),
                jaccard=inter/(k+l-inter),expected_f1=2*k*l/(n*(k+l)),expected_jaccard=expected_j)


def read_release(path):
    value=io.read(path)
    io.require(path==ROOT/"results"/RUN/"evaluate_release.json"
               and value["run_id"]==RUN and value["released"] is True,"one score release required")
    recipe,registered_parent=fit_run.validate(io.verified(value["fit_release"]))
    io.require(value["source_hashes"]==recipe["source_hashes"],"fit and score source manifests differ")
    parent=io.read(io.verified(value["parent_evaluation"]))
    io.require(recipe["parent_evaluation"]==value["parent_evaluation"] and parent==registered_parent,
               "parent release differs")
    io.require(value["fit_terminal_verified"] is True,"fit termination verification required")
    fit=io.read(io.verified(value["fit_summary"]))
    io.require(fit["release_ref"]==value["fit_release"] and fit["status"] in {"completed","wave2_held"}
               and fit["val_access"] is False and fit["test_access"] is False,"fit summary differs")
    declared={row["id"]:row for row in recipe["jobs"]}
    outcomes={}
    io.require(set(value["fit_refs"])==set(declared)==set(fit["jobs"]), "terminal evidence incomplete")
    for name,ref in value["fit_refs"].items():
        io.require(ref["path"]==f"results/{RUN}/{name}/summary.json" and ref==fit["jobs"][name]["summary_ref"],
                   "terminal path or identity differs")
        result=io.read(io.verified(ref))
        aggregate_receipt=fit["jobs"][name]
        status=result["status"]
        data=result["result"]
        io.require(result["run_id"]==RUN and result["release_ref"]==value["fit_release"]
                   and result["job_id"]==name and result["job"]==declared[name]
                   and status==aggregate_receipt["status"] and status in {"completed","scientific_stopped","held"}
                   and data["valid_endpoint"]==aggregate_receipt["valid_endpoint"]==(status=="completed"),
                   "fit terminal provenance or status differs")
        if status=="completed":
            io.require(data["attempts"]==data["actual_endpoint"]==data["fixed_endpoint"]==128
                       and fit_run.MIN_ACCEPTED<=data["accepted"]<=128
                       and data["terminal_checkpoint"]==data["checkpoints"]["128"]
                       and data["checkpoints"]["128"]["path"]==f"artifacts/{RUN}/fit/{name}/checkpoint_128.pt",
                       "fixed endpoint differs")
            io.verified(data["checkpoints"]["128"])
        outcomes[name]=result
    return value,recipe,parent,outcomes


def evaluate(path):
    release,recipe,archive,outcomes=read_release(path)
    local=ROOT/"artifacts"/RUN/"evaluate01"
    public=ROOT/"results"/RUN/"evaluate01"
    io.require(not local.exists() and not public.exists(),"single val score already claimed")
    local.mkdir(parents=True); public.mkdir(parents=True)
    ref=io.identity(path)
    io.write(local/"started.json",dict(release_ref=ref,started_utc=io.utc()))
    t=perf_counter()
    try:
        torch.set_num_threads(1)
        anchor=io.read(io.verified(archive["release"]["input_refs"]["anchor_analysis"]))
        raw,constants=matched.load_train({"anchor_analysis":anchor})
        del raw
        events=load_val(constants)
        identity=matched.baseline.data_identity(events)
        io.require(identity==archive["result"]["val_identity"],"original val identity differs")
        groups=measured.rollout_groups(events)
        output_steps=sum(count*len(cohort) for count,cohort in groups.items())
        originals=restore(archive)
        parents={}
        for seed in (24111,24121,24131):
            key=f"graph_{seed}"
            model=originals[key]
            score=archive["result"]["val"][key]
            measured.warm_up(model)
            timing=measured.time_archived_endpoint(model,groups)
            parents[str(seed)]={**aggregate(score["score"]),"checkpoint_ref":score["checkpoint_ref"],
                                "spectrum":score["spectrum"],"inference":timing}
        results={}
        supports={}
        for job in recipe["jobs"]:
            key=job["id"]
            endpoint=outcomes[key]
            data=endpoint["result"]
            if not data["valid_endpoint"]:
                results[key]=dict(status="not_scored",training_status=endpoint["status"],
                                  attempts=data["attempts"],accepted=data["accepted"])
                continue
            model=refit_model(endpoint,release["fit_release"])
            measured.warm_up(model)
            clock=measured.InferenceClock(model)
            with patch.object(model,"initialize",clock.initialize),patch.object(measured.scoring,"integrate",clock.integrate):
                scored=graph_run.score(model,events,constants,"val",local/key,{"work":{}})
            result=aggregate(scored)
            io.require(scored["data_identity"]==identity,"scored split identity differs")
            supports[key]=set(map(tuple,model.edge_index.T[model.c.detach()>0].tolist()))
            result.update(status="scored",complexity=model.complexity(),
                          spectrum=endpoint_spectrum(model,local/key),inference=clock.summary(output_steps),
                          checkpoint_ref=data["checkpoints"]["128"])
            results[key]=result
        full_checks={str(seed):full_guard(parents[str(seed)],results.get(f"full_{seed}"),recipe["full_retention_guard"])
                     for seed in (24111,24121,24131)}
        overlap={}
        for variant in {job["variant"] for job in recipe["jobs"]}:
            overlap[variant]={}
            for i,j in ((24111,24121),(24111,24131),(24121,24131)):
                a,b=f"{variant}_{i}",f"{variant}_{j}"
                if a in supports and b in supports:
                    overlap[variant][f"{i}/{j}"]=hypergeom_agreement(supports[a],supports[b])
        support_ref=matched.baseline.atomic_write(local/"supports.json",{k:sorted(v) for k,v in supports.items()})
        summary=dict(run_id=RUN,status="completed",release_ref=ref,parents=parents,val=results,
                     full_guards=full_checks,support_overlap=overlap,support_ref=support_ref,val_identity=identity,
                     new_fom=0,test_trajectory_access=False,completed_utc=io.utc(),wall_s=perf_counter()-t)
        summary_ref=matched.baseline.atomic_write(public/"summary.json",summary,public=True)
        io.write(local/"completed.json",dict(status="completed",summary_ref=summary_ref))
        print(summary_ref,flush=True)
    except BaseException as error:
        io.write(local/"failed.json",dict(status="failed",error=repr(error),release_ref=ref))
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release",required=True,type=Path)
    evaluate(parser.parse_args().release.resolve())
