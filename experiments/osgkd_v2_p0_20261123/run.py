"""Released P0 diagnostics: no training, integration, or prediction-array access."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import sys
from time import perf_counter, process_time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from kiki.models.osgkd import OSGKD, OSGKDConfig
from kiki.data.osgkd import load_fit297
from kiki.evaluation.gftd_analysis import quotient_basis

RUN = "osgkd_v2_p0_20261123"
JOB = "probe01"
MANIFEST = "results/osgkd_fit297_20261113/artifacts.json"
SOURCES = (f"experiments/{RUN}/run.py", "src/kiki/models/osgkd.py",
           "src/kiki/data/osgkd.py", "src/kiki/evaluation/osgkd.py",
           "src/kiki/evaluation/gftd_analysis.py", "experiments/evaluate_l1_latent.py", MANIFEST)


def relative_path(value):
    if not isinstance(value, str):
        raise ValueError("canonical repository-relative path required")
    p = PurePosixPath(value)
    if "\\" in value or ":" in value or p.is_absolute() or any(
            part in ("..", ".") for part in value.split("/")) or not value or "//" in value:
        raise ValueError("canonical repository-relative path required")
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("path escapes repository")
    return path


def identity(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        size = stream.tell()
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": digest, "bytes": size}


def verified(ref):
    path = relative_path(ref["path"])
    if identity(path) != ref:
        raise ValueError("input identity differs: " + ref["path"])
    return path


def write_json(path, value):
    raw = (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(raw)
    return identity(path)


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"count": 0, "quantiles": None}
    if not np.isfinite(values).all():
        raise ValueError("nonfinite diagnostic")
    return {"count": int(values.size), "mean": float(values.mean()),
            "quantiles": dict(zip(("min", "q05", "q25", "median", "q75", "q95", "max"),
                                  np.quantile(values, [0, .05, .25, .5, .75, .95, 1]).tolist()))}


def spectrum(model):
    n = model.n
    if n != 10:
        raise ValueError("ten-node model required")
    x0 = torch.cat((model.theta_ref, torch.zeros(4*n, dtype=torch.float64)))
    scale = torch.cat((torch.ones(n, dtype=torch.float64), model.omega_scale, model.init_bounds))
    def function(y):
        return model.rhs(None, x0 + scale*y) / scale
    zero = torch.zeros(5*n, dtype=torch.float64)
    rhs = model.rhs(None, x0).detach().numpy()
    if not np.isfinite(rhs).all():
        return {"qualified": False, "direction": "unresolved", "reason": "nonfinite fixed-reference RHS",
                "rhs_max_abs": None}, {"rhs": None}
    equilibrium_residual = float(np.max(abs(rhs)))
    if equilibrium_residual > 1e-10:
        return {"qualified": False, "reason": "fixed reference is not an equilibrium",
                "rhs_max_abs": equilibrium_residual}, {"rhs": rhs.tolist()}
    jac = torch.autograd.functional.jacobian(function, zero).detach().numpy()
    helmert = quotient_basis(n)
    basis = np.zeros((5*n, 5*n-1))
    basis[:n, :n-1] = helmert
    basis[n:, n-1:] = np.eye(4*n)
    common = np.r_[np.ones(n)/np.sqrt(n), np.zeros(4*n)]
    checks = {"basis_orthogonality": float(np.max(abs(basis.T@basis-np.eye(5*n-1)))),
              "basis_common_angle": float(np.max(abs(basis.T@common))),
              "common_angle_relative_residual": float(np.linalg.norm(jac@common)/max(np.linalg.norm(jac), 1e-300))}
    fd = []
    reduced = basis.T@jac@basis
    for h in (1e-5, 5e-6):
        cols = []
        for direction in np.eye(5*n):
            delta = torch.from_numpy(h*direction)
            cols.append(((function(delta)-function(-delta))/(2*h)).detach().numpy())
        candidate = np.column_stack(cols)
        fd.append({"h": h, "relative_frobenius": float(np.linalg.norm(candidate-jac)/max(np.linalg.norm(jac), 1e-300)),
                   "max_real": float(np.linalg.eigvals(basis.T@candidate@basis).real.max())})
    qualified = all(v <= 1e-10 for v in checks.values()) and all(x["relative_frobenius"] <= 1e-6 for x in fd)
    poles, vectors = np.linalg.eig(reduced)
    order = np.argsort(-poles.real)
    residuals = [float(np.linalg.norm(reduced@vectors[:,i]-poles[i]*vectors[:,i]) /
                       max((np.linalg.norm(reduced)+abs(poles[i]))*np.linalg.norm(vectors[:,i]), 1e-300)) for i in order]
    index = int(order[0])
    vector = basis@vectors[:, index]
    weights = model.mass.detach().numpy().diagonal()
    weights = weights/weights.sum()
    speed = vector[n:2*n]
    physical_speed = speed * model.omega_scale.detach().numpy()
    common_speed = physical_speed@weights
    common_normalized = common_speed/model.omega_scale.detach().numpy()
    components = {"relative_angle": vector[:n], "relative_speed": speed-common_normalized,
                  "common_speed": common_normalized, "p": vector[2*n:3*n],
                  "a": vector[3*n:4*n], "e": vector[4*n:]}
    norms = {key: float(np.vdot(value,value).real) for key,value in components.items()}
    maximum = float(poles.real.max())
    error = max(abs(x["max_real"]-maximum) for x in fd)
    signs = [np.sign(maximum), *[np.sign(x["max_real"]) for x in fd]]
    direction = ("positive" if maximum > 0 else "negative") if qualified and len(set(signs)) == 1 and abs(maximum) > max(1e-7,error) else "unresolved"
    summary = {"qualified": bool(qualified), "rhs_max_abs": equilibrium_residual, **checks,
               "finite_difference": fd, "dimension": 49, "max_real": maximum,
               "direction": direction, "max_real_fd_difference": error,
               "eigenpair_relative_residual_max": max(residuals),
               "dominant_component_squared_norms": norms,
               "component_definition": "normalized coordinates; inertia common physical speed; speed components are not orthogonal and do not define additive shares",
               "dominant_block_absolute_squared_shares": {key: float(np.vdot(vector[j*n:(j+1)*n],vector[j*n:(j+1)*n]).real / np.vdot(vector,vector).real) for j,key in enumerate(("angle", "speed", "p", "a", "e"))}}
    detail = {"normalized_jacobian": jac.tolist(), "quotient_jacobian": reduced.tolist(),
              "basis": basis.tolist(), "scales": scale.tolist(), "reference": x0.tolist(),
              "eigenvalues": [{"real": float(poles[i].real), "imag": float(poles[i].imag), "relative_residual": residuals[j]} for j,i in enumerate(order)],
              "dominant_vector_real": vector.real.tolist(), "dominant_vector_imag": vector.imag.tolist()}
    return summary, detail


def trigger_status(item, endpoint):
    if item is None:
        return "missing"
    classification = item["classification"]
    category = classification["category"]
    if category in ("numerical_unresolved", "latency_uncovered"):
        return category
    if item.get("los_pair_pass") is not True:
        return "numerical_unresolved"
    if classification["predicted_los"]:
        time = item["predicted_time_record_s"]
        if time is None:
            return "missing"
        return "in_true_window" if time <= endpoint+1e-12 else "outside_true_window"
    return "no_trigger"


def analyze_initial(model, events, rows):
    init_wall, init_cpu = perf_counter(), process_time()
    with torch.no_grad():
        theta = torch.tensor(np.stack([e["history"]["theta"] for e in events]), dtype=torch.float64)
        omega = torch.tensor(np.stack([e["history"]["omega"] for e in events]), dtype=torch.float64)
        initial = model.initialize(theta, omega)
    init_cost = {"wall_s":perf_counter()-init_wall,"cpu_s":process_time()-init_cpu,"events":len(events),"calls":1}
    if not torch.equal(initial[:,:10],theta[:,-1]) or not torch.equal(initial[:,10:20],omega[:,-1]):
        raise ValueError("initializer changed motion endpoint")
    ratio = (initial[:,20:]/model.init_bounds).numpy()
    if not np.isfinite(ratio).all() or np.max(abs(ratio)) > 1+1e-12:
        raise ValueError("invalid initial latent bounds")
    records = []
    for i,(event,row) in enumerate(zip(events,rows,strict=True)):
        endpoint = .1+float(event["t"][-1])
        actual = trigger_status(row.get("classification_actual"),endpoint)
        eight = trigger_status(row.get("classification_8s"),endpoint)
        scope = row["scopes"]["1"]
        metric = scope["metrics"]
        records.append({"event_id":event["id"], "label":event["label"], "endpoint_record_s":endpoint,
                        "actual_future_points":len(event["t"])-1,
                        "actual_status":actual, "eight_status":eight,
                        "actual_classification":row.get("classification_actual"),
                        "eight_classification":row.get("classification_8s"),
                        "full_1s":bool(scope["full_horizon_truth"]),
                        "mean_motion_squared_error":None if metric is None else metric["mean_motion_error"],
                        "normalized_latents":ratio[i].tolist(),
                        "saturation":{str(threshold): {key:np.flatnonzero(abs(ratio[i,j*10:(j+1)*10])>=threshold).tolist() for j,key in enumerate(("p","a","e"))} for threshold in (.99,.95)},
                        "tanh_derivative":(1-ratio[i]**2).tolist()})
    cohorts = {"all":list(range(len(events))), "stable":[i for i,e in enumerate(events) if e["label"]=="stable"],
               "unstable":[i for i,e in enumerate(events) if e["label"]=="unstable"],
               "detected":[i for i,r in enumerate(records) if r["label"]=="unstable" and r["actual_status"]=="in_true_window"],
               "missed":[i for i,r in enumerate(records) if r["label"]=="unstable" and r["actual_status"]=="no_trigger"]}
    summary = {}
    for cohort,indices in cohorts.items():
        item = {"denominator":len(indices), "actual_status_counts":dict(Counter(records[i]["actual_status"] for i in indices)),
                "eight_status_counts":dict(Counter(records[i]["eight_status"] for i in indices)), "latent":{}}
        for j,key in enumerate(("p","a","e")):
            r = ratio[indices,j*10:(j+1)*10]
            item["latent"][key] = {"absolute_ratio":distribution(abs(r).ravel()), "tanh_derivative":distribution((1-r*r).ravel()),
                "thresholds":{str(t):{"saturated_components":int((abs(r)>=t).sum()),
                                     "events_any":int((abs(r)>=t).any(axis=1).sum()),
                                     "per_machine_counts":(abs(r)>=t).sum(axis=0).tolist()} for t in (.99,.95)}}
        item["error_pairing"] = {}
        for full in (True,False):
            subset = [i for i in indices if records[i]["full_1s"]==full]
            good = [i for i in subset if records[i]["mean_motion_squared_error"] is not None]
            item["error_pairing"]["complete_1s" if full else "shorter_than_1s"] = {
                "denominator":len(subset), "missing":len(subset)-len(good),
                "motion_squared_error":distribution([records[i]["mean_motion_squared_error"] for i in good]),
                "by_saturation_099":{str(sat):{"count":len(selected := [i for i in good if bool((abs(ratio[i])>=.99).any())==sat]),
                    "motion_squared_error":distribution([records[i]["mean_motion_squared_error"] for i in selected])} for sat in (False,True)}}
        summary[cohort] = item
    summary["initialization_cost"] = init_cost
    summary["false_alarm_8s"] = {"denominator":sum(e["label"]=="stable" for e in events),
        "count":sum(r["label"]=="stable" and r["eight_status"]=="in_true_window" for r in records),
        "unresolved_or_missing":sum(r["label"]=="stable" and r["eight_status"] in
          ("missing","numerical_unresolved","latency_uncovered") for r in records)}
    return summary, records


def validate_rows(rows, events):
    if not isinstance(rows,list) or len(rows)!=297 or len({r["event_id"] for r in rows})!=297:
        raise ValueError("exact rows cohort required")
    mapping = {r["event_id"]:r for r in rows}
    if set(mapping)!={e["id"] for e in events}:
        raise ValueError("row identities differ")
    ordered = [mapping[e["id"]] for e in events]
    for event,row in zip(events,ordered,strict=True):
        if row["group"]!=event["group"] or row["label"]!=event["label"] or row["actual_future_points"]!=len(event["t"])-1:
            raise ValueError("row descriptors or coverage differ")
    return ordered


def execute(release_path):
    started, cpu = perf_counter(), process_time()
    torch.set_num_threads(1)
    release_path = release_path.resolve()
    if not release_path.is_relative_to(ROOT):
        raise ValueError("release must be inside repository")
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if (release.get("run_id"),release.get("job_id"),release.get("released")) != (RUN,JOB,True):
        raise ValueError("unreleased or wrong job")
    if not set(SOURCES)<=set(release["source_hashes"]):
        raise ValueError("missing source hashes")
    for name,digest in release["source_hashes"].items():
        if identity(relative_path(name))["sha256"]!=digest:
            raise ValueError("source identity differs: "+name)
    manifest = json.loads((ROOT/MANIFEST).read_text(encoding="utf-8"))
    if release["audit_ref"]["path"] != "results/osgkd_fit297_20261113/port_audit_release.json":
        raise ValueError("fixed v1 audit release required")
    files = {ref["path"]:ref for ref in manifest["files"]}
    for variant,old_name in (("C0","C0"),("C1","C1_dynamic")):
        if release["models"][variant] != manifest["selected_models"][old_name]:
            raise ValueError("model differs from fixed v1 selection")
        row_path = f"artifacts/osgkd_fit297_20261113/eval01/{old_name}/rows.json"
        if release["rows"][variant] != files[row_path]:
            raise ValueError("rows differ from fixed v1 evaluation")
    verified(release["audit_ref"])
    paths = {}
    for family in ("models","rows"):
        if set(release[family])!={"C0","C1"}:
            raise ValueError("C0/C1 input pair required")
        paths[family] = {variant:verified(ref) for variant,ref in release[family].items()}
    local = ROOT/"artifacts"/RUN/JOB
    public = ROOT/"results"/RUN/JOB
    if local.exists() or public.exists():
        raise FileExistsError("job output already exists; no overwrite or replay")
    local.mkdir(parents=True)
    public.mkdir(parents=True)
    inputs = {"release":identity(release_path), "release_content":release}
    write_json(local/"started.json", {"status":"started", "run_id":RUN,"job_id":JOB,
               "pid":os.getpid(),"started_utc":datetime.now(timezone.utc).isoformat(),
               "command":[sys.executable,*sys.argv],"inputs":inputs})
    try:
        events,constants = load_fit297(release["audit_ref"])
        if len(events)!=297 or Counter(e["label"] for e in events)!={"stable":267,"unstable":30}:
            raise ValueError("exact train cohort required")
        summaries,details = {},{}
        for variant in ("C0","C1"):
            saved = torch.load(paths["models"][variant],map_location="cpu",weights_only=True)
            if saved["attempt"]!=1500 or saved["model"]["config"]["variant"]!=variant:
                raise ValueError("checkpoint branch or update count differs")
            if saved["provenance"]["audit_ref"]!=release["audit_ref"]:
                raise ValueError("checkpoint observation provenance differs")
            model = OSGKD.from_checkpoint(saved["model"]).cpu().double().eval()
            rows = validate_rows(json.loads(paths["rows"][variant].read_text(encoding="utf-8")),events)
            spectral,full = spectrum(model)
            initial,records = analyze_initial(model,events,rows)
            summaries[variant] = {"spectrum":spectral,"initialization_and_classification":initial}
            details[variant] = records
            write_json(local/f"{variant}_spectrum.json",full)
            write_json(local/f"{variant}_events.json",records)
            del saved,model,rows
        common = [i for i in range(297) if all(details[v][i]["label"]=="unstable" and
                  details[v][i]["actual_classification"] is not None and
                  details[v][i]["actual_classification"]["timing_comparable"] and
                  details[v][i]["actual_status"]=="in_true_window" for v in ("C0","C1"))]
        timing = {"common_comparable_denominator":len(common), "models":{}}
        for v in ("C0","C1"):
            timing["models"][v] = distribution([details[v][i]["actual_classification"]["trigger_time_error_s"] for i in common])
        write_json(local/"paired_timing.json",{"event_ids":[events[i]["id"] for i in common],"summary":timing})
        outputs = [identity(path) for path in sorted(local.glob("*.json"))]
        summary = {"run_id":RUN,"job_id":JOB,"status":"completed","inputs":inputs,"models":summaries,
                   "paired_timing":timing, "data_identity":{"constants_id":constants["constants_id"],
                   "ordered_ids_sha256":hashlib.sha256(json.dumps([e["id"] for e in events],ensure_ascii=False,separators=(",",":")).encode()).hexdigest()},
                   "work":{"initialized_events":594,"new_integrations":0,"training_updates":0,"new_fom":0},
                   "environment":{"python":sys.version,"numpy":np.__version__,"torch":str(torch.__version__),
                                  "platform":platform.platform(),"device":"cpu","dtype":"float64","threads":torch.get_num_threads(),"command":[sys.executable,*sys.argv]},
                   "cost":{"wall_s":perf_counter()-started,"cpu_s":process_time()-cpu,"scope":"verification, bundle load and diagnostics; final serialization excluded"},
                   "private_outputs":outputs}
        summary_ref = write_json(public/"summary.json",summary)
        write_json(local/"completed.json",{"status":"completed","summary":summary_ref,
                   "outputs":outputs,"wall_s":perf_counter()-started,"cpu_s":process_time()-cpu})
    except Exception as exc:
        write_json(local/"failed.json",{"status":"failed","error_type":type(exc).__name__,"error":str(exc),
                   "wall_s":perf_counter()-started,"cpu_s":process_time()-cpu})
        raise


def self_check():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    for variant in ("C0","C1"):
        model = OSGKD(OSGKDConfig(variant),mass=np.ones(10),damping=np.ones(10),F=np.eye(10),
                      Sp=np.ones(10),Se=np.ones(10),omega_scale=np.linspace(.8,1.2,10),
                      theta_ref=np.linspace(0,.2,10),edge_index=np.array([np.arange(9),np.arange(1,10)]))
        result,detail = spectrum(model)
        assert result["qualified"] and len(detail["eigenvalues"])==49
        theta = torch.zeros((2,11,10),dtype=torch.float64)
        initial = model.initialize(theta,theta)
        assert initial.shape==(2,50) and torch.equal(initial[:,:20],torch.zeros((2,20),dtype=torch.float64))
        events = [{"id":str(i),"group":"synthetic","label":"stable" if i<267 else "unstable",
                   "t":np.arange(101)*.01,"history":{"theta":np.zeros((11,10)),"omega":np.zeros((11,10))}}
                  for i in range(297)]
        no_hit = {"classification":{"category":"window_no_los_unsettled","predicted_los":False},
                  "los_pair_pass":True,"predicted_time_record_s":None}
        rows = [{"event_id":e["id"],"group":e["group"],"label":e["label"],"actual_future_points":100,
                 "classification_actual":no_hit,"classification_8s":no_hit,
                 "scopes":{"1":{"full_horizon_truth":True,"metrics":{"mean_motion_error":.1}}}}
                for e in events]
        rows = validate_rows(rows[::-1],events)
        summaries,records = analyze_initial(model,events,rows)
        assert summaries["stable"]["denominator"]==267 and summaries["missed"]["denominator"]==30
        assert summaries["initialization_cost"]["calls"]==1 and len(records)==297
        assert summaries["all"]["error_pairing"]["complete_1s"]["denominator"]==297
        original_rhs = model.rhs
        model.rhs = lambda t,x: torch.full_like(x,float("nan"))
        unresolved,detail = spectrum(model)
        assert unresolved["rhs_max_abs"] is None and detail["rhs"] is None
        json.dumps([unresolved,detail],allow_nan=False)
        model.rhs = original_rhs
    hit = {"classification":{"category":"predicted_los","predicted_los":True},"los_pair_pass":True,"predicted_time_record_s":.5}
    assert trigger_status(hit,.5)=="in_true_window" and trigger_status(hit,.4)=="outside_true_window"
    assert trigger_status(None,1)=="missing"
    for value in (None,1,"../x","C:/x","a\\b","/x","a/./b"):
        try:
            relative_path(value)
        except ValueError:
            pass
        else:
            raise AssertionError(value)
    print("synthetic self-check passed; private assets were not opened")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--release",type=Path)
    group.add_argument("--self-check",action="store_true")
    args = parser.parse_args()
    self_check() if args.self_check else execute(args.release)
