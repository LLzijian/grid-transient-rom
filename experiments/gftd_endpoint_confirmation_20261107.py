"""New-direction confirmation of the frozen two-stage endpoint initializer."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import scipy
from scipy.stats import beta

from gftd_initialization_20261107 import (
    ROOT, digest, endpoint as hard_endpoint, functional, infer_endpoint as infer_simple,
    validate_observation, write_json,
)
from gftd_endpoint_denoising_20261107 import angle_basis, infer_endpoint as infer_smooth
from gftd_noise_confirmation_20261107 import evaluate_numerical, refit_eligible, score_task

OUTPUT = ROOT/"results/gftd_endpoint_confirmation_20261107"
PROTOCOL_SHA256 = "4bba5ceb4a5e33c5a80bf8fbeb9d9d5158b896a0687d3f70db73d6f96d01a2f0"
SOURCE_IDS = {str(case):f"input_{33*case:03d}" for case in range(4)}
TERMINAL = ("pass","prediction_fail","numerically_unresolved","optimization_no_output","blocked_by_stage1")


def checked_references(manifest):
    if manifest["case_to_source_input"] != SOURCE_IDS:
        raise ValueError("noiseless case-to-source identity changed")
    for name in ("truth","noiseless_observations","noiseless_summary"):
        source = manifest[name]
        if digest(ROOT/source["path"]) != source["sha256"]:
            raise ValueError(f"reference changed: {name}")
    return manifest


def generate_observations(manifest, output):
    """Truth is local to generation; fitting receives only the observation file."""
    checked_references(manifest)
    truth = json.loads((ROOT/manifest["truth"]["path"]).read_text(encoding="utf-8"))
    noiseless = json.loads((ROOT/manifest["noiseless_observations"]["path"]).read_text(encoding="utf-8"))
    old_summary = json.loads((ROOT/manifest["noiseless_summary"]["path"]).read_text(encoding="utf-8"))
    old_rows = {row["input_id"]:row for row in old_summary["rows"]}
    observed, plan = {}, []
    for case in range(4):
        source_id = SOURCE_IDS[str(case)]
        history = np.asarray(truth[source_id]["history"],dtype=np.float64)
        base = validate_observation(noiseless[source_id])
        old = old_rows[source_id]
        if old["case"]!=case or old["sigma"]!=0. or old["classification"]!="pass":
            raise ValueError("noiseless output reference is missing or unqualified")
        if not truth[source_id]["reference_numerical"]["passed"] or not truth[source_id]["endpoint_qualification"]["passed"]:
            raise ValueError("reference numerical qualification failed")
        if case==0 and any(old["main"][step]["joint_rms"]>1e-8 for step in ("future_main","future_fine")):
            raise ValueError("noiseless equilibrium reference failed")
        if float(base["sigma"]) != 0. or history.shape != (11,15) or not np.isfinite(history).all():
            raise ValueError("missing or inconsistent noiseless reference")
        if not (np.array_equal(base["theta_history"],history[:,:3])
                and np.array_equal(base["omega_history"],history[:,3:6])
                and np.array_equal(base["p_end"],history[-1,6:9])):
            raise ValueError("noiseless observations disagree with referenced history")
        key = f"input_{len(observed):03d}"
        observed[key] = noiseless[source_id]
        plan.append(dict(input_id=key,case=case,rep=None,sigma=0.,seed=None,reference_input_id=source_id))
        for rep in range(16):
            seed = 130000+100*case+rep
            noise = np.random.Generator(np.random.PCG64(seed)).standard_normal((11,6))
            for sigma in (1e-6,1e-4):
                key = f"input_{len(observed):03d}"
                measurement = history[:,:6]+.1*sigma*noise
                observed[key] = dict(theta_history=measurement[:,:3],omega_history=measurement[:,3:6],
                    p_end=history[-1,6:9].copy(),history_times=np.linspace(-.1,0.,11),sigma=sigma)
                plan.append(dict(input_id=key,case=case,rep=rep,sigma=sigma,seed=seed,reference_input_id=source_id))
    write_json(Path(output)/"observations.json",observed)
    write_json(Path(output)/"input_plan.json",plan)
    write_json(Path(output)/"reference_manifest.json",manifest)
    del truth, noiseless, history, old_summary, old_rows
    return Path(output)/"observations.json",Path(output)/"input_plan.json"


def numerical_pass(check):
    if check is None or check.get("passed") is not True:
        return False
    roundtrip = np.asarray(check.get("roundtrip_scaled_max",[]),dtype=np.float64)
    values = np.asarray([check.get(name,np.inf) for name in (
        "history_main_fine_rms","future_main_fine_rms","future_main_fine_peak")])
    return bool(roundtrip.shape==(2,) and np.isfinite(roundtrip).all() and np.all(roundtrip<=1e-7)
                and np.isfinite(values).all() and np.all(values<=[1e-7,1e-5,1e-4]))


def supply_eligible(task, check, observation):
    if task is None or task["status"] not in ("converged","budget_limited") or task["best"] is None:
        return False
    try:
        expected = hard_endpoint(validate_observation(observation),task["best"]["u"],False)
        return numerical_pass(check) and np.array_equal(expected,task["best"]["state"])
    except (ValueError,TypeError):
        return False


def fit_packages(observed, output=None):
    f = functional(.5)
    records = {key:dict(status="planned",stage1=None,stage1_numerical=None,stage2=None,
        stage2_numerical=None,fine=None,fine_numerical=None,final_point=None) for key in sorted(observed)}
    def checkpoint(stage):
        if output is not None:
            temporary = Path(output)/"fits.partial.tmp.json"
            write_json(temporary,dict(stage=stage,records=records))
            temporary.replace(Path(output)/"fits.partial.json")
    checkpoint("planned")
    for key,record in records.items():
        obs = validate_observation(observed[key])
        if float(obs["sigma"])==0.:
            record["status"]="reused_noiseless"
        else:
            task = infer_simple(obs,f,amplitude_free=False,initial=np.zeros(3))
            record["stage1"] = task
            record["stage1_numerical"],record["stage1_numerical_error"] = evaluate_numerical(f,task)
            if not supply_eligible(task,record["stage1_numerical"],obs):
                record["status"]="blocked_by_stage1"
            else:
                record["stage2"] = infer_smooth(obs,f,task["best"]["u"],smooth=True)
                record["stage2_numerical"],record["stage2_numerical_error"] = evaluate_numerical(f,record["stage2"])
                record["status"]="completed"
                record["final_point"]="main"
        checkpoint("main_fits")
        print(json.dumps(dict(input_id=key,status=record["status"],calls={name:None if record[name] is None
            else record[name]["actual_residual_calls"] for name in ("stage1","stage2")})),flush=True)
    requests = [(row["stage2_numerical"]["history_main_fine_rms"],key) for key,row in records.items()
                if row["stage2"] is not None and row["stage2"]["status"]!="failed"
                and row["stage2"]["objective_guard_passed"]
                and refit_eligible(row["stage2_numerical"])]
    requests.sort(key=lambda item:(-item[0],item[1]))
    for _,key in requests[:4]:
        row = records[key]
        row["fine"] = infer_smooth(observed[key],f,row["stage1"]["best"]["u"],smooth=True,
            initial=row["stage2"]["best"]["u"],step=.00125)
        row["fine_numerical"],row["fine_numerical_error"] = evaluate_numerical(f,row["fine"])
        row["final_point"] = "fine"
        checkpoint("fine_refits")
    return dict(records=records,Q=angle_basis(f.mass),requested_fine_refits=len(requests),
                performed_fine_refits=min(4,len(requests)))


def score_class(task, check, score, reference):
    if task is None or task["best"] is None:
        return "optimization_no_output"
    if (task["status"]=="failed" or not numerical_pass(check) or score is None or "error" in score
            or not reference["reference_numerical"]["passed"] or not reference["endpoint_qualification"]["passed"]):
        return "numerically_unresolved"
    if np.any(np.asarray(score["p_constraint_error"])!=0) or score["future_main_pass"]!=score["future_fine_pass"]:
        return "numerically_unresolved"
    return "pass" if score["future_fine_pass"] else "prediction_fail"


def complete_class(record, stage1_class, scores, reference):
    if record["status"]=="blocked_by_stage1" or stage1_class not in ("pass","prediction_fail"):
        return "blocked_by_stage1"
    main = record["stage2"]
    final_name = record["final_point"]
    final_task = record["fine"] if final_name=="fine" else main
    final_check = record["fine_numerical"] if final_name=="fine" else record["stage2_numerical"]
    outcome = score_class(final_task,final_check,scores[final_name],reference)
    if outcome not in ("pass","prediction_fail"):
        return "numerically_unresolved" if final_name=="fine" else outcome
    if main is None or main["status"]=="failed" or not main["objective_guard_passed"] or not final_task["objective_guard_passed"]:
        return "numerically_unresolved"
    if final_name=="fine":
        if not refit_eligible(record["stage2_numerical"]) or scores["main"] is None or "error" in scores["main"]:
            return "numerically_unresolved"
        if np.any(np.asarray(scores["main"]["p_constraint_error"])!=0):
            return "numerically_unresolved"
        outcomes = [scores[name][field] for name in ("main","fine") for field in ("future_main_pass","future_fine_pass")]
        if len(set(outcomes))!=1:
            return "numerically_unresolved"
    return outcome


def statistics(rows, field):
    layers=[]
    for case in range(4):
        for sigma in (1e-6,1e-4):
            layer=[row for row in rows if row["case"]==case and row["sigma"]==sigma]
            reps=[row["rep"] for row in layer]
            if len(reps)!=len(set(reps)) or any(rep not in range(16) for rep in reps):
                raise ValueError("invalid or duplicate noise direction")
            counts={label:sum(row[field]==label for row in layer) for label in TERMINAL+("planned","not_run")}
            if sum(counts.values())!=len(layer):
                raise ValueError("unregistered result classification")
            complete=set(reps)==set(range(16)) and all(row[field] in TERMINAL for row in layer)
            k=counts["pass"]
            interval=None if not complete else [0. if k==0 else float(beta.ppf(.025,k,17-k)),
                                                1. if k==16 else float(beta.ppf(.975,k+1,16-k))]
            layers.append(dict(case=case,sigma=sigma,denominator=16,counts=counts,successes=k,complete=complete,
                               cp95=interval,simultaneous_failure_upper95=None))
    all_complete=all(layer["complete"] for layer in layers)
    if all_complete:
        for layer in layers:
            failures=16-layer["successes"]
            layer["simultaneous_failure_upper95"]=1. if failures==16 else float(beta.ppf(1-.05/8,failures+1,16-failures))
    return dict(family="complete_process_eight_strata_primary" if field=="complete_classification" else "stage1_eight_strata_separate_secondary",
                simultaneous_bound_available=all_complete,layers=layers,
                joint_sixteen_strata_coverage_claim=False)


def score_frozen(fits_path, fits_hash, obs_path, plan_path, manifest):
    if digest(fits_path)!=fits_hash:
        raise ValueError("fits changed before truth scoring")
    fits=json.loads(Path(fits_path).read_text(encoding="utf-8"))
    if digest(obs_path)!=fits["observations_sha256"] or digest(plan_path)!=fits["plan_sha256"]:
        raise ValueError("frozen observations or plan changed")
    checked_references(manifest)
    truth=json.loads((ROOT/manifest["truth"]["path"]).read_text(encoding="utf-8"))
    prior=json.loads((ROOT/manifest["noiseless_summary"]["path"]).read_text(encoding="utf-8"))
    old_rows={row["input_id"]:row for row in prior["rows"]}
    obs=json.loads(Path(obs_path).read_text(encoding="utf-8"))
    plan=json.loads(Path(plan_path).read_text(encoding="utf-8"))
    f=functional(.5)
    rows=[]
    for entry in plan:
        key=entry["input_id"]
        row=dict(entry,stage1_classification="not_run",complete_classification="not_run")
        record=fits["records"].get(key)
        if record is None or record["status"]=="planned":
            row["stage1_classification"]=row["complete_classification"]="planned" if record else "not_run"
        elif record["status"]=="reused_noiseless":
            old=old_rows[entry["reference_input_id"]]
            if old["sigma"]!=0. or old["case"]!=entry["case"]:
                raise ValueError("noiseless record identity mismatch")
            row.update(reused_parent_record=old,stage1_classification=old["classification"],
                       complete_classification=old["classification"],reused_exactly=True)
        else:
            observation=validate_observation(obs[key])
            reference=truth[entry["reference_input_id"]]
            scores={name:score_task(f,record[name],observation,reference) for name in ("stage1","stage2","fine")}
            for score in scores.values():
                if score is not None and "error" not in score:
                    score["endpoint_estimation_scaled_error"]=score.pop("endpoint_observation_scaled_noise")
            one=score_class(record["stage1"],record["stage1_numerical"],scores["stage1"],reference)
            if one in ("pass","prediction_fail") and not supply_eligible(record["stage1"],record["stage1_numerical"],observation):
                one="numerically_unresolved"
            two=complete_class(record,one,{"main":scores["stage2"],"fine":scores["fine"]},reference)
            row.update(stage1_classification=one,complete_classification=two,scores=scores,final_point=record["final_point"],
                       stage1_status=record["stage1"]["status"],stage2_status=None if record["stage2"] is None else record["stage2"]["status"])
            final_task=record["fine"] if record["final_point"]=="fine" else record["stage2"]
            if final_task is not None and final_task["best"] is not None:
                state=final_task["best"]["state"]
                correction=np.asarray(state)[:6]-np.r_[observation["theta_history"][-1],observation["omega_history"][-1]]
                row["endpoint_correction"]=correction
                row["common_angle_correction"]=float(f.coi_weights@correction[:3])
        rows.append(row)
    comparisons=[]
    for row in rows:
        if row["sigma"]==0.:
            continue
        score=row.get("scores",{})
        final=score.get("fine" if row.get("final_point")=="fine" else "stage2")
        first=score.get("stage1")
        delta=None if first is None or final is None or "error" in first or "error" in final else (
            final["future_fine"]["joint_rms"]-first["future_fine"]["joint_rms"])
        label="unavailable" if delta is None else "same" if abs(delta)<=1e-10 else "improved" if delta<0 else "worsened"
        comparisons.append(dict(input_id=row["input_id"],difference=delta,classification=label,
            regression=row["stage1_classification"]=="pass" and row["complete_classification"]!="pass"))
    worst={}
    for stage in ("stage1","complete"):
        values=[]
        for row in rows:
            if row["sigma"]==0.:
                continue
            name="stage1" if stage=="stage1" else "fine" if row.get("final_point")=="fine" else "stage2"
            score=row.get("scores",{}).get(name)
            if score is not None and "error" not in score:
                values.append(score["future_fine"])
        worst[stage]=dict(denominator=128,scored_outputs=len(values),
            worst_rms=None if not values else max(metric["joint_rms"] for metric in values),
            worst_peak=None if not values else max(metric["peak"] for metric in values))
    return dict(rows=rows,statistics_complete=statistics(rows,"complete_classification"),worst_errors=worst,
                statistics_stage1=statistics(rows,"stage1_classification"),comparisons=comparisons,
                comparison_counts={label:sum(row["classification"]==label for row in comparisons)
                                   for label in ("same","improved","worsened","unavailable")},
                independent_units=64,paired_noise_inputs=128,scope="new directions at four exposed states")


def main():
    if digest(ROOT/"reports/gftd_endpoint_confirmation_20261107_protocol.md")!=PROTOCOL_SHA256:
        raise ValueError("confirmation protocol changed")
    if np.__version__!="2.5.2" or scipy.__version__!="1.18.0":
        raise ValueError("registered library versions required")
    release=json.loads((OUTPUT/"execution_release.json").read_text(encoding="utf-8-sig"))
    if release.get("released") is not True:
        raise PermissionError("root confirmation release required")
    for path,expected in release["source_sha256"].items():
        if digest(ROOT/path)!=expected:
            raise ValueError(f"frozen source changed: {path}")
    manifest=checked_references(release["reference_sources"])
    with (OUTPUT/"started.json").open("x",encoding="utf-8") as stream:
        json.dump(dict(action_id=release["action_id"],status="RUNNING"),stream)
    begin=time.perf_counter()
    receipt=dict(action_id=release["action_id"],status="FAILED",numpy=np.__version__,scipy=scipy.__version__)
    try:
        obs_path,plan_path=generate_observations(manifest,OUTPUT)
        obs_hash,plan_hash=digest(obs_path),digest(plan_path)
        observed=json.loads(obs_path.read_text(encoding="utf-8"))
        if len(observed)!=132:
            raise ValueError("fixed confirmation input scope changed")
        fits=fit_packages(observed,OUTPUT)
        fits.update(observations_sha256=obs_hash,plan_sha256=plan_hash,source_sha256=release["source_sha256"],reference_sources=manifest)
        fits_path=OUTPUT/"fits.json"
        write_json(fits_path,fits)
        fits_hash=digest(fits_path)
        write_json(OUTPUT/"fits_frozen.json",dict(fits_sha256=fits_hash,observations_sha256=obs_hash,plan_sha256=plan_hash))
        result=score_frozen(fits_path,fits_hash,obs_path,plan_path,manifest)
        write_json(OUTPUT/"summary.json",result)
        tasks=[task for row in fits["records"].values() for name in ("stage1","stage2","fine") if (task:=row[name]) is not None]
        if len(tasks)>260 or sum(task["actual_residual_calls"] for task in tasks)>235200:
            raise AssertionError("confirmation finite workload exceeded")
        receipt.update(status="COMPLETED",fits_sha256=fits_hash,summary_sha256=digest(OUTPUT/"summary.json"),
                       optimization_tasks=len(tasks),actual_residual_calls=sum(task["actual_residual_calls"] for task in tasks))
    except Exception as exc:
        receipt["error"]=repr(exc)
        raise
    finally:
        receipt.update(wall_seconds=time.perf_counter()-begin,real_data_reads=False,
            source_unchanged=all(digest(ROOT/path)==sha for path,sha in release["source_sha256"].items()))
        write_json(OUTPUT/"execution_receipt.json",receipt)
        print(json.dumps(receipt),flush=True)


if __name__=="__main__":
    main()
