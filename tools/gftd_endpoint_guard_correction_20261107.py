"""Append-only, same-point objective-guard audit. No model evaluations."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/"results/gftd_endpoint_guard_correction_20261107"
DESIGN = "reports/gftd_endpoint_denoising_20261107_guard_repair_design.md"
DESIGN_SHA256 = "a66a3f15919233a5b6f89867063eddd5a0e21b4858ba406d893699779e34051f"
CHILD = "results/gftd_endpoint_denoising_20261107"
PARENT = "results/gftd_noise_confirmation_20261107"
ORIGINAL_FITS_SHA = "38416c6e88cfc660418c79de735a0d4853662d39e692376e05c3ec818517877b"
ORIGINAL_SUMMARY_SHA = "c5f8b17e45ef861cb1d912795e2e10016b375d0432507055adf283e8b6710a29"
EXPECTED_IMPACT = {(name,"anchor") for name in ("input_020","input_023","input_026","input_043")}
COMMON_SOURCES = (
    "experiments/gftd_initialization_20261107.py", "experiments/gftd_observability_20261107.py",
    "src/kiki/physics/kuramoto_functional.py", "src/kiki/physics/graph.py",
)
EVIDENCE_KEYS = {"known_original_fits","known_original_summary","common_sources","same_fixed_parameter_factory",
                 "same_observations","parent_fits_bound","versions","reviewed_arithmetic_paths"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def binary64(value):
    if isinstance(value,list):
        items = [binary64(item) for item in value]
        if not items or any(shape != items[0][0] for shape,data in items):
            raise ValueError("empty or ragged numeric array")
        return (len(items),)+items[0][0], b"".join(data for shape,data in items)
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError("finite binary64 values required")
    return (), struct.pack(">d",float(value))


def same_bits(left, right, shape=None):
    try:
        a,b = binary64(left),binary64(right)
        return a == b and (shape is None or a[0] == shape)
    except (ValueError,OverflowError,TypeError):
        return False


def identity_certificate(task, parent, obs, parent_obs, group, evidence):
    """Future-blind algebraic predicate, applied uniformly to every main task."""
    best, old = task.get("best") or {}, parent.get("best") or {}
    keys = {"theta_history","omega_history","history_times","p_end","sigma"}
    shapes = {"theta_history":(11,3),"omega_history":(11,3),"history_times":(11,),"p_end":(3,),"sigma":()}
    observed_identity = set(obs) == keys and set(parent_obs) == keys and all(
        same_bits(obs[key],parent_obs[key],shapes[key]) for key in keys)
    coordinate = False
    if group == "anchor":
        coordinate = all(same_bits(task.get(name),old.get("u"),(3,))
                         for name in ("initial","feasible_parent")) and same_bits(best.get("u"),old.get("u"),(3,))
    elif group == "smooth":
        vectors = [best.get("u"),task.get("initial"),task.get("feasible_parent")]
        coordinate = all(isinstance(u,list) and len(u)==8 and all(
            isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and v==0. for v in u[:5])
            and same_bits(u[5:],old.get("u"),(3,)) for u in vectors)
    loss = best.get("observation_loss")
    checks = dict(source_and_execution_identity=set(evidence)==EVIDENCE_KEYS and all(value is True for value in evidence.values()),
        observations_identical=observed_identity,
        state_identical=same_bits(best.get("state"),old.get("state"),(15,)),
        coordinates_identical=coordinate,
        history_step_identical=same_bits(task.get("history_step"),parent.get("history_step"),()),
        scale_identical=same_bits(task.get("optimizer_residual_divisor"),parent.get("optimizer_residual_divisor"),()),
        same_step_losses_identical=same_bits(loss,task.get("parent_same_step_loss"),())
            and same_bits(loss,task.get("initial_same_step_loss"),())
            and same_bits(old.get("observation_loss"),old.get("observation_loss"),()),
        same_step_guard=task.get("objective_guard_passed") is True,
        parent_success_rescaled_path=parent.get("status")=="converged",
        nonfailed_finite_execution=task.get("status") in ("converged","budget_limited")
            and parent.get("status") in ("converged","budget_limited")
            and task.get("nonfinite_evaluations")==0 and parent.get("nonfinite_evaluations")==0,
        no_fine_step_replacement=same_bits(task.get("history_step"),.0025,()))
    proven = all(checks.values())
    facts = None
    if same_bits(loss,loss,()) and same_bits(old.get("observation_loss"),old.get("observation_loss"),()):
        parent_loss = float(old["observation_loss"])
        child_loss = float(loss)
        facts = dict(child_loss=child_loss,parent_loss=parent_loss,
                     child_bits=binary64(child_loss)[1].hex(),parent_bits=binary64(parent_loss)[1].hex(),
                     difference=child_loss-parent_loss,
                     difference_in_parent_ulps=(child_loss-parent_loss)/math.ulp(parent_loss),
                     ulp_is_admission_condition=False)
    return dict(same_point_certificate=proven,checks=checks,
                failed_identity_conditions=[name for name,passed in checks.items() if not passed],
                arithmetic_facts=facts,
                cross_path_guard_applicability="redundant_identical_point" if proven else "identity_not_proven",
                original_frozen_parent_objective_guard=task.get("frozen_parent_objective_guard"),
                effective_guard=task.get("objective_guard_passed") if proven else task.get("frozen_parent_objective_guard"))


def numerical_qualification(record, summary_group, reference=None):
    check = record.get("numerical") or {}
    pair = check.get("roundtrip_scaled_max",[])
    values = [check.get(name) for name in ("history_main_fine_rms","future_main_fine_rms","future_main_fine_peak")]
    finite = len(pair)==2 and all(isinstance(value,(int,float)) and math.isfinite(value) for value in pair+values)
    reference = reference or {}
    passed = bool(finite and reference.get("endpoint_qualification",{}).get("passed") is True
                  and reference.get("reference_numerical",{}).get("passed") is True
                  and check.get("passed") is True and not record.get("numerical_error")
                  and max(pair)<=1e-7 and values[0]<=1e-7 and values[1]<=1e-5 and values[2]<=1e-4)
    score = (summary_group.get("scored") or {}).get("main")
    if not passed or score is None or "error" in score or record.get("fine") is not None:
        return dict(passed=False,classification="numerically_unresolved")
    errors = score.get("p_constraint_error")
    if not isinstance(errors,list) or len(errors)!=3 or any(not math.isfinite(v) or v!=0. for v in errors):
        return dict(passed=False,classification="numerically_unresolved")
    predictions = []
    for name in ("future_main","future_fine"):
        metric = score[name]
        if not all(isinstance(metric[key],(int,float)) and math.isfinite(metric[key]) for key in ("joint_rms","peak")):
            return dict(passed=False,classification="numerically_unresolved")
        predictions.append(metric["joint_rms"]<=1e-3 and metric["peak"]<=1e-2)
        if score.get(name+"_pass") is not predictions[-1]:
            return dict(passed=False,classification="numerically_unresolved")
    if predictions[0] != predictions[1]:
        return dict(passed=False,classification="numerically_unresolved")
    return dict(passed=True,classification="pass" if predictions[0] else "prediction_fail")


def aggregate(summary, adoption=None):
    if adoption is None:
        sys.path.insert(0,str(ROOT/"experiments"))
        from gftd_endpoint_denoising_20261107 import adoption_conditions
        adoption = adoption_conditions
    rows = summary["rows"]
    for name,step,refined in (("adoption_main","future_main",False),("adoption_fine","future_fine",False),
                              ("adoption_refined_main","future_main",True),("adoption_refined_fine","future_fine",True)):
        summary[name] = adoption(rows,step,use_refit=refined)
    results = [summary[name] for name in ("adoption_main","adoption_fine","adoption_refined_main","adoption_refined_fine")]
    summary["adoption_consistent"] = all(value["conditions"]==results[0]["conditions"] for value in results)
    summary["worth_independent_confirmation"] = (all(value["meets_all"] for value in results)
        and summary["adoption_consistent"] and summary["noiseless_equilibrium_calibration"])
    return summary


def audit(child_fits, parent_fits, original_summary, obs, parent_obs, evidence, expected_impact=EXPECTED_IMPACT,
          adoption=None, expected_main_count=256, references=None):
    corrected = copy.deepcopy(original_summary)
    row_index = {row["input_id"]:row for row in corrected["rows"]}
    certificates, changes, redundant_false_guards = [], [], []
    for key,groups in child_fits["records"].items():
        for group,record in groups.items():
            task = record.get("main")
            if task is None:
                continue
            certificate = identity_certificate(task,parent_fits["records"][key]["main"],obs[key],parent_obs[key],group,evidence)
            entry = dict(input_id=key,group=group,certificate=certificate,qualification=None)
            if task.get("frozen_parent_objective_guard") is False and certificate["same_point_certificate"]:
                redundant_false_guards.append((key,group))
                entry["qualification"] = numerical_qualification(record,row_index[key][group],(references or {}).get(key))
                if entry["qualification"]["passed"]:
                    previous = row_index[key][group]["classification"]
                    effective = entry["qualification"]["classification"]
                    if previous != effective:
                        row_index[key][group]["classification"] = effective
                        changes.append(dict(input_id=key,group=group,original=previous,effective=effective))
            certificates.append(entry)
    changed = {(row["input_id"],row["group"]) for row in changes}
    if len(certificates)!=expected_main_count:
        raise ValueError("main scan count differs from frozen scope")
    if set(redundant_false_guards) != set(expected_impact):
        raise ValueError(f"identity impact requires independent review: {sorted(redundant_false_guards)}")
    if changed != set(expected_impact):
        raise ValueError(f"correction impact requires independent review: {sorted(changed)}")
    aggregate(corrected,adoption)
    # Only the proven classification paths and their adoption aggregates may change.
    restored = copy.deepcopy(corrected)
    for change in changes:
        row = next(row for row in restored["rows"] if row["input_id"]==change["input_id"])
        row[change["group"]]["classification"] = change["original"]
    dependent = {"adoption_main","adoption_fine","adoption_refined_main","adoption_refined_fine",
                 "adoption_consistent","worth_independent_confirmation"}
    for name in dependent:
        restored[name] = original_summary[name]
    if restored != original_summary:
        raise AssertionError("nonclassification evidence changed")
    for name in ("adoption_main","adoption_fine","adoption_refined_main","adoption_refined_fine"):
        if corrected[name]["high_noise_means"] != original_summary[name]["high_noise_means"]:
            raise AssertionError("classification-independent means changed")
    counts = {group:{label:sum(row[group]["classification"]==label for row in corrected["rows"])
                     for label in ("pass","prediction_fail","numerically_unresolved","optimization_no_output","planned","not_run")}
              for group in ("parent","anchor","smooth")}
    paired = {}
    for baseline in ("parent","anchor"):
        outcomes = []
        for row in corrected["rows"]:
            before,after = row[baseline]["classification"],row["smooth"]["classification"]
            if before not in ("pass","prediction_fail") or after not in ("pass","prediction_fail"):
                label = "unresolved"
            elif before == after:
                label = "both_pass" if after=="pass" else "both_fail"
            else:
                label = "repaired" if after=="pass" else "regressed"
            outcomes.append(dict(input_id=row["input_id"],classification=label))
        paired[baseline] = dict(rows=outcomes,counts={label:sum(row["classification"]==label for row in outcomes)
            for label in ("both_pass","both_fail","repaired","regressed","unresolved")})
    return dict(scan_count=len(certificates),certificates=certificates,changes=changes,summary=corrected,
                classification_counts=counts,paired_prediction_outcomes=paired,
                original_worth_independent_confirmation=original_summary["worth_independent_confirmation"],
                unchanged_nonclassification_evidence=True)


def functional_call_signature(source):
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="fit_packages")
    calls = [node for node in ast.walk(function) if isinstance(node,ast.Call)
             and isinstance(node.func,ast.Name) and node.func.id=="functional"]
    return [ast.dump(call,include_attributes=False) for call in calls]


def load_evidence(root):
    child_release = read(root/CHILD/"execution_release.json")
    parent_release = read(root/PARENT/"execution_release.json")
    hashes = {}
    for directory,release in ((CHILD,child_release),(PARENT,parent_release)):
        if release.get("released") is not True:
            raise ValueError("original action was not released")
        for path,expected in release["source_sha256"].items():
            if digest(root/path)!=expected:
                raise ValueError(f"frozen source or evidence changed: {path}")
            hashes[path] = expected
        receipt = read(root/directory/"execution_receipt.json")
        if receipt["status"]!="COMPLETED" or receipt["source_unchanged"] is not True:
            raise ValueError("original execution identity incomplete")
        for name in ("fits","summary"):
            path = f"{directory}/{name}.json"
            actual = digest(root/path)
            if actual != receipt[name+"_sha256"]:
                raise ValueError("original result hash does not match its receipt")
            hashes[path] = actual
        for name in ("execution_release.json","execution_receipt.json"):
            path = f"{directory}/{name}"
            hashes[path] = digest(root/path)
    child = read(root/CHILD/"fits.json")
    parent = read(root/PARENT/"fits.json")
    observed_hash = digest(root/PARENT/"observations.json")
    receipts = [read(root/directory/"execution_receipt.json") for directory in (CHILD,PARENT)]
    call_signatures = [functional_call_signature((root/path).read_text(encoding="utf-8")) for path in (
        "experiments/gftd_endpoint_denoising_20261107.py","experiments/gftd_noise_confirmation_20261107.py")]
    evidence = dict(
        known_original_fits=hashes[f"{CHILD}/fits.json"]==ORIGINAL_FITS_SHA,
        known_original_summary=hashes[f"{CHILD}/summary.json"]==ORIGINAL_SUMMARY_SHA,
        common_sources=all(child_release["source_sha256"][path]==parent_release["source_sha256"][path]
                           for path in COMMON_SOURCES),
        same_fixed_parameter_factory=call_signatures[0]==call_signatures[1] and len(call_signatures[0])==1,
        same_observations=child["input_sha256"]["observations"]==parent["observations_sha256"]==observed_hash,
        parent_fits_bound=child["input_sha256"]["parent_fits"]==digest(root/PARENT/"fits.json"),
        versions=all(receipt["numpy"]=="2.5.2" and receipt["scipy"]=="1.18.0" for receipt in receipts),
        reviewed_arithmetic_paths=True)
    return child,parent,read(root/CHILD/"summary.json"),read(root/PARENT/"observations.json"),evidence,hashes


def main():
    if digest(ROOT/DESIGN)!=DESIGN_SHA256:
        raise ValueError("guard correction design changed")
    release = read(OUTPUT/"execution_release.json")
    if release.get("released") is not True:
        raise PermissionError("root correction release required")
    for path,expected in release["source_sha256"].items():
        if digest(ROOT/path)!=expected:
            raise ValueError(f"correction source changed: {path}")
    with (OUTPUT/"started.json").open("x",encoding="utf-8") as stream:
        json.dump(dict(action_id=release["action_id"],status="RUNNING"),stream)
    begin = time.perf_counter()
    receipt = dict(action_id=release["action_id"],status="FAILED",technical_repair_round=1,
                   science_integrations=0,science_optimizations=0)
    original_hashes = {}
    try:
        child,parent,summary,obs,evidence,original_hashes = load_evidence(ROOT)
        reference_source = read(ROOT/PARENT/"isolated_truth.json")
        references = {key:{name:value[name] for name in ("endpoint_qualification","reference_numerical")}
                      for key,value in reference_source.items()}
        del reference_source
        result = audit(child,parent,summary,obs,copy.deepcopy(obs),evidence,references=references)
        if not all(digest(ROOT/path)==sha for path,sha in original_hashes.items()):
            raise ValueError("original evidence changed during correction")
        certificates = dict(source_sha256=original_hashes,evidence=evidence,scan_count=result["scan_count"],
            certificates=result["certificates"],changes=result["changes"],
            arithmetic_path_note="Parent success path reconstructs raw=result.fun*scale; child uses direct raw. Original residual vectors were not saved. Per-element 1-ULP provenance was not replayed.")
        write(OUTPUT/"certificates.json",certificates)
        write(OUTPUT/"effective_classification_appendix.json",dict(
            original_summary_sha256=ORIGINAL_SUMMARY_SHA,original_worth_independent_confirmation=result["original_worth_independent_confirmation"],
            changes=result["changes"],unchanged_nonclassification_evidence=True,effective_summary=result["summary"],
            classification_counts=result["classification_counts"],paired_prediction_outcomes=result["paired_prediction_outcomes"],
            confirmation_released=False))
        receipt.update(status="COMPLETED",scan_count=result["scan_count"],changed_classifications=len(result["changes"]),
                       certificates_sha256=digest(OUTPUT/"certificates.json"),
                       appendix_sha256=digest(OUTPUT/"effective_classification_appendix.json"),
                       original_worth_independent_confirmation=result["original_worth_independent_confirmation"],
                       effective_worth_independent_confirmation=result["summary"]["worth_independent_confirmation"])
    except Exception as exc:
        receipt["error"] = repr(exc)
        raise
    finally:
        receipt.update(wall_seconds=time.perf_counter()-begin,
            original_artifacts_unchanged=all(digest(ROOT/path)==sha for path,sha in original_hashes.items()),
            correction_sources_unchanged=all(digest(ROOT/path)==sha for path,sha in release["source_sha256"].items()))
        write(OUTPUT/"execution_receipt.json",receipt)
        print(json.dumps(receipt),flush=True)


if __name__ == "__main__":
    main()
