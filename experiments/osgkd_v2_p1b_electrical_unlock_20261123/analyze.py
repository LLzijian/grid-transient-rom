"""Score the preregistered high-order-potential contrast from saved endpoints."""
from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import sys
from time import perf_counter, process_time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import numpy as np
import torch
from experiments.osgkd_v2_conflict_profile_20261123.run import signed_error
from experiments.osgkd_v2_early_closure_20261123.run import identity, read, require, verified, write
from experiments.osgkd_v2_early_electrical_20261123.analyze import early_aggregate, evaluate_prefixes, practical_gates
from kiki.data.osgkd import load_fit297
from kiki.evaluation.osgkd import summarize
from kiki.models.osgkd import OSGKD
from kiki.models.osgkd_anchored_potential import OSGKDAnchoredPotential

RUN = "osgkd_v2_p1b_electrical_unlock_20261123"
LOCAL = ROOT / "artifacts" / RUN / "pair01"
PUBLIC = ROOT / "results" / RUN
FIELDS = ("theta", "omega", "tau_m", "tau_e")


def input_endpoints(receipt, clean):
    chosen = clean["choice"]
    parent = clean["inputs"][chosen]
    require(parent["checkpoint"] == receipt["release"]["checkpoint_ref"] and clean["gates"][chosen]["passed"],
            "working parent differs")
    inputs = {"parent": {"checkpoint": parent["checkpoint"], "rows": parent["rows"],
                         "predictions": parent["predictions"], "summary": parent["summary"]}}
    for treatment, bundle in receipt["result"]["treatments"].items():
        directory = LOCAL / treatment / "evaluation"
        inputs[treatment] = {"checkpoint": bundle["checkpoint"], "rows": identity(directory / "rows.json"),
                             "predictions": identity(directory / "predictions.npz"), "summary": bundle["evaluation_summary_ref"]}
    return inputs


def validate_profiles(events, archives, missed):
    counts = {name: {"denominator":0,"complete_10_nodes_0.03":0,"complete_10_nodes_capped_0.25":0,
                     "full_0.25_s":0} for name in ("train_all","parent_misses")}
    checked = 0
    for index,event in enumerate(events):
        times = np.asarray(event["t"])
        require(times.ndim == 1 and len(times) >= 4 and np.isfinite(times).all()
                and abs(times[0]) <= 1e-10 and np.allclose(np.diff(times),.01,rtol=0,atol=1e-9),
                "actual future grid differs")
        stop = min(25,len(times)-1)
        masks = [np.asarray(event["masks"][key]) for key in FIELDS]
        require(all(mask.dtype == np.bool_ and mask.shape == (len(times),10)
                    and np.array_equal(mask,np.isfinite(event["targets"][key]))
                    for key,mask in zip(FIELDS,masks,strict=True)),
                "event mask coverage differs")
        in_scope = event["id"] in missed
        for name in (("train_all","parent_misses") if in_scope else ("train_all",)):
            count = counts[name]
            count["denominator"] += 1
            count["full_0.25_s"] += stop == 25
            count["complete_10_nodes_0.03"] += all(mask[:4].all() for mask in masks)
            count["complete_10_nodes_capped_0.25"] += all(mask[:stop+1].all() for mask in masks)
        if in_scope:
            observed = np.r_[event["history"]["theta"][-1],event["history"]["omega"][-1]]
            for file in archives.values():
                path = file[f"event_{index:04d}_fine"][:stop+1]
                require(path.shape == (stop+1,50) and np.array_equal(path[0,:20],observed),
                        "saved fine trajectory leaves the observed initial state or window")
                checked += 1
    return {"observations":counts,"saved_anchor_checks":checked}


def signed_profile(events, archives, models, profile):
    selected = {r["event_id"]:r["selected_machine"] for r in profile["signed_electrical"]
                if all(x["persistent_80pct"] is True for x in r["endpoints"].values())}
    require(len(selected) == 22, "frozen persistent electric-error sample differs")
    candidate = models["candidate"].checkpoint()
    candidate["version"] = OSGKD.checkpoint_version
    candidate["state_dict"].pop("rho")
    frozen_shared = OSGKD.from_checkpoint(candidate).eval()
    signed = []
    for index, event in enumerate(events):
        event_id = event["id"]
        if event_id not in selected:
            continue
        steps = min(25, len(event["t"])-1)
        i = selected[event_id]
        predictions = {name: file[f"event_{index:04d}_fine"][:steps+1] for name,file in archives.items()}
        scores = {name: signed_error(event, model, predictions[name], i) for name,model in models.items()}
        with torch.no_grad():
            state = torch.as_tensor(predictions["candidate"][1:], dtype=torch.float64)
            component = (models["candidate"].ports(state).Fc - frozen_shared.ports(state).Fc) / models["candidate"].Se
            mask = event["masks"]["tau_e"][1:steps+1, i]
            extra = float(component[:, i].numpy()[mask].mean()) if mask.any() else None
        signed.append({"event_id":event_id,"machine":i,"signed_electric":scores,"candidate_added_coupling_mean":extra})
    return signed


def summarize_gates(scores, early, full_rows):
    report = practical_gates(scores, early, full_rows)
    def stable8(name):
        scope = scores[name]["splits"]["train_all"]["stable_8s"]["official"]
        return scope["motion_risk_v1"] if scope else None
    candidate, control = stable8("candidate"), stable8("control")
    report["checks"]["stable_8s_risk"] = candidate is not None and control is not None and candidate < 1. and candidate <= 1.05 * control
    hits = {name: set(s["classification"]["train_all"]["actual"]["detected_unstable_ids"]) for name,s in scores.items()}
    report["new_hits_over_parent"] = sorted(hits["candidate"] - hits["parent"])
    report["new_hits_over_control"] = sorted(hits["candidate"] - hits["control"])
    report["pipeline_motion_and_stability_passed"] = all(report["checks"].values())
    report["detection_gate"] = all(len(hits["candidate"]) >= len(hits[name]) + 2 for name in ("parent","control"))
    report["passed"] = report["pipeline_motion_and_stability_passed"] and report["detection_gate"]
    return report


def summarize_profiles(full_rows, cohorts, signed):
    stable = {}
    for group in ("worsened", "remaining"):
        ids = cohorts[group]
        stable[group] = {}
        for name, rows in full_rows.items():
            included = [rows[event]["scopes"]["8"]["metrics"] for event in sorted(ids)]
            values = [metric["mean_motion_error"] for metric in included if metric is not None]
            stable[group][name] = {"covered":len(values),"denominator":len(ids),
                                  "mean_motion_error":float(np.mean(values)) if len(values) == len(ids) else None}
    electrical = {}
    for name in ("parent", "control", "candidate"):
        metrics = [r["signed_electric"][name] for r in signed if r["signed_electric"][name]["mean"] is not None]
        electrical[name] = {"covered":len(metrics),"denominator":len(signed),
                            "mean_signed_error":float(np.mean([m["mean"] for m in metrics])) if metrics else None,
                            "persistent_events":sum(m["persistent_80pct"] for m in metrics)}
    new_force = [r["candidate_added_coupling_mean"] for r in signed if r["candidate_added_coupling_mean"] is not None]
    return {"stable":stable,"electrical":electrical,"candidate_added_coupling_mean":float(np.mean(new_force)) if new_force else None}


def main():
    started, cpu = perf_counter(), process_time()
    torch.set_num_threads(1)
    marker = json.loads((LOCAL / "completed.json").read_bytes())
    receipt = read(marker["summary_ref"])
    require(receipt["run_id"] == RUN and receipt["status"] == "completed" and receipt["result"]["fixed_endpoint"] == 2020,
            "completed fixed endpoint required")
    release = read(receipt["release_ref"])
    require(release == receipt["release"], "release content differs")
    for name,digest in release["source_hashes"].items():
        require(identity(ROOT / name)["sha256"] == digest, "source differs: " + name)
    refs = release["input_refs"]
    clean = read(refs["clean_analysis"])
    profile_summary = read(refs["conflict_summary"])
    profile = read(profile_summary["result"]["rows_ref"])
    events, constants = load_fit297(refs["audit"])
    cohorts = {k:set(v) for k,v in profile["cohort_event_ids"].items()}
    require({k:len(v) for k,v in cohorts.items()} == {"worsened":24,"remaining":243,"parent_misses":27},
            "frozen cohort sizes differ")
    inputs = input_endpoints(receipt, clean)
    models, archives, full_rows, scores = {}, {}, {}, {}
    with ExitStack() as stack:
        for name,bundle in inputs.items():
            saved = torch.load(verified(bundle["checkpoint"]), map_location="cpu", weights_only=True)
            require(saved["attempt"] == (1860 if name == "parent" else 2020), "checkpoint endpoint differs")
            cls = OSGKDAnchoredPotential if name == "candidate" else OSGKD
            models[name] = cls.from_checkpoint(saved["model"]).eval().requires_grad_(False)
            archives[name] = stack.enter_context(np.load(verified(bundle["predictions"]), allow_pickle=False))
            rows = read(bundle["rows"])
            require([row["event_id"] for row in rows] == [e["id"] for e in events], "evaluation identity/order differs")
            full_rows[name] = {r["event_id"]:r for r in rows}
            scores[name] = summarize(rows, constants)
            stored = read(bundle["summary"])
            require(scores[name] == {k:stored[k] for k in scores[name]}, "endpoint summary differs")
        coverage = validate_profiles(events, archives, cohorts["parent_misses"])
        early_rows, work = evaluate_prefixes(events, constants, models, archives)
        signed = signed_profile(events, archives, models, profile)
    early = early_aggregate(early_rows, {"parent_misses":cohorts["parent_misses"]})
    rows_ref = write(LOCAL / "early_metric_rows.json", {"scopes":early_rows,"persistent_sign_profile":signed})
    write(PUBLIC / "analysis.json",
          {"run_id":RUN,"job":marker["summary_ref"],"inputs":inputs,"early":early,"official_scores":scores,
           "fixed_subgroup_profiles":summarize_profiles(full_rows,cohorts,signed),"early_rows_ref":rows_ref,
           "short_window_coverage":coverage,
           "practical_gates":summarize_gates(scores,early,full_rows),"port_state_points":work,
           "cost":{"wall_s":perf_counter()-started,"cpu_s":process_time()-cpu}})


if __name__ == "__main__":
    main()
