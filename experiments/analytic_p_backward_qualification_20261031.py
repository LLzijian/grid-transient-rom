"""Explicit analytic_p_backward_overlay_v1 qualification. No optimizer or fit.

evaluate --candidate A96_analytic (or A16_analytic) evaluates at most once.
summary requires both complete receipts. Keep wrapper logs outside PRIVATE.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
from types import MethodType

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"experiments"),str(ROOT/"src")]
import p_reverse_precision_20261031 as reverse
import checkpoint_selection_20261030 as previous

PRIVATE=ROOT/"private_data/analytic_p_backward_qualification_20261031"
PUBLIC=ROOT/"results/analytic_p_backward_qualification_20261031"
PARENT=ROOT/"private_data/p_reverse_precision_20261031"
FORMAT="analytic_p_backward_overlay_v1"
PARENT_COMMIT="be39cfb5f84c079753d3bd250efc94e8a6177ed5"
CANDIDATES={"A96_analytic":"A96","A16_analytic":"step000016"}
PARENT_SHA={"A96":"ec8c78cbff0b6dd65bd3ba6dfc5a0ed65e7fa418227c90adc3a703311aebadf4","step000016":"3b637077976f035fd7cb6d042893a9095e61b0b51a424ed1b77f911eb91d5e5c"}
sha,read,ident,write_once,safe=reverse.sha,reverse.read,reverse.ident,reverse.write_once,reverse.safe


def source_closure():
    names=(Path(__file__).name,"p_reverse_precision_20261031.py","force_graph_model_20261028.py","state_gated_memory_model_20261028.py","memory_transport_model_20261029.py","state_gated_memory_study_20261028.py","checkpoint_selection_20261030.py",
        "memory_transport_study_20261029.py","checkpoint_selection_decision_20261030.py","p_port_diagnostic_20261031.py","closure_probe_20261030.py","force_exchange_study_20261028.py",
        "evaluate_l1_latent.py","evaluate_long_latent.py","evaluate_structured_rom.py","fixed_edge_pair_20261019.py","summarize_train_local_memory_0098.py","train_structured_rom.py","train_local_memory_0098.py","active_fom_pair_fit_20261020.py","frequency_dissipation_20261021.py","closure_discovery_confirmation.py")
    return {name:sha(ROOT/"experiments"/name) for name in names}


def state_identity(model):
    return reverse.state_identity(model)


def assert_only_buffer(base,candidate):
    a,b=state_identity(base),state_identity(candidate)
    if set(a)!=set(b) or any(a[name]!=b[name] for name in a if name!="p_backward"):
        raise ValueError("candidate changes state beyond p_backward")
    if {n:p.requires_grad for n,p in base.named_parameters()}!={n:p.requires_grad for n,p in candidate.named_parameters()} or base.training!=candidate.training:
        raise ValueError("candidate changes train/gradient settings")
    return {"only_patch_key":"p_backward","base_states":a,"candidate_states":b,"flags_unchanged":True}


def analytic_reference(q):
    """Unique analytic80 construction, checked at120 and by independent expm."""
    import mpmath as mp
    exp80,analytic80=reverse.high_precision_reference(q,80)
    exp120,analytic120=reverse.high_precision_reference(q,120)
    with mp.workdps(120):
        scale=max(mp.mpf(1),*(abs(v) for row in exp120 for v in row))
        diffs=[max(abs(a-b) for ra,rb in zip(rows,exp120) for a,b in zip(ra,rb)) for rows in (analytic80,analytic120,exp80)]
        qualified=all(d<=mp.mpf("1e-60")*scale for d in diffs)
        n=len(analytic80);matrix=np.zeros((n,3*n),dtype=np.float64)
        for i,row in enumerate(analytic80):matrix[i,[i,n+i,2*n+i]]=[float(v) for v in row]
        receipt={"Q_sha256":reverse.array_hash(q),"B_ref_sha256":reverse.array_hash(matrix),"B_ref":matrix.tolist(),"format":FORMAT,"analytic_implementation_sha256":sha(Path(__file__)),
            "high_precision_qualified":bool(qualified),"analytic80_analytic120_expm80_vs_expm120_maxabs":[mp.nstr(v,125) for v in diffs],"construction":"analytic80 rounded once to binary64; exact binary64 Q input"}
    from scipy.linalg import expm
    scipy_ref=expm(q)[:len(matrix)]
    receipt["float64_independent_gate_passed"]=bool(np.allclose(matrix,scipy_ref,atol=1e-12,rtol=1e-12))
    receipt["float64_independent_maxabs"]=float(np.max(np.abs(matrix-scipy_ref)))
    receipt["qualified"]=receipt["high_precision_qualified"] and receipt["float64_independent_gate_passed"]
    return matrix,receipt


def attach_overlay(base,payload):
    candidate=copy.deepcopy(base)
    reference=np.asarray(payload["B_ref"],dtype=np.float64)
    if reference.shape!=tuple(base.p_backward.shape) or not np.isfinite(reference).all() or reverse.array_hash(reference)!=payload["B_ref_sha256"]:
        raise ValueError("invalid explicit analytic buffer")
    with torch.no_grad():candidate.p_backward.copy_(torch.as_tensor(reference,dtype=torch.float64))
    candidate.clear_cache()
    proof=assert_only_buffer(base,candidate)
    if proof["candidate_states"]!=payload["candidate_state_identity"]:
        raise ValueError("overlay state identity differs")
    candidate._analytic_overlay=copy.deepcopy(payload)
    candidate.export=MethodType(export_overlay,candidate)
    return candidate


def export_overlay(model,path):
    if model._cache is not None:raise ValueError("close graph cache before overlay export")
    payload=model._analytic_overlay
    if state_identity(model)!=payload["candidate_state_identity"]:raise ValueError("candidate no longer matches immutable overlay")
    write_once(path,payload)
    return payload


def strict_base_loader(label,path):
    import closure_probe_20261030 as probe
    spec=previous.model_spec(label)
    if Path(spec["export"]).resolve()!=Path(path).resolve():raise ValueError("parent is outside fixed allowlist")
    return probe.load_model(spec)


def load_overlay(path):
    payload=read(path)
    if payload.get("format")!=FORMAT or payload.get("patch_key")!="p_backward" or payload.get("candidate_id") not in CANDIDATES:
        raise ValueError("dedicated analytic overlay format required")
    label=CANDIDATES[payload["candidate_id"]]
    if payload["parent_label"]!=label or payload["parent"]["sha256"]!=PARENT_SHA[label]:raise ValueError("parent allowlist identity differs")
    parent_path=ROOT/payload["parent"]["path"]
    if sha(parent_path)!=payload["parent"]["sha256"] or read(parent_path)["format"]!=payload["parent_format"]:
        raise ValueError("frozen parent changed")
    if payload["source_closure"]!=source_closure():raise ValueError("overlay implementation source changed")
    if sha(ROOT/payload["protocol"]["path"])!=payload["protocol"]["sha256"]:raise ValueError("overlay protocol changed")
    math_path=ROOT/payload["math_receipt"]["path"]
    if sha(math_path)!=payload["math_receipt"]["sha256"]:raise ValueError("math receipt changed")
    receipt=read(math_path)
    if not receipt["qualified"] or receipt["analytic_implementation_sha256"]!=sha(Path(__file__)) or receipt["B_ref_sha256"]!=payload["B_ref_sha256"] or receipt["B_ref"]!=payload["B_ref"]:
        raise ValueError("analytic math provenance differs")
    # Original loader executes all original assertions before any patch.
    base=strict_base_loader(label,parent_path)
    if reverse.tensor_hash(base.p_backward)!=payload["original_B_sha256"] or state_identity(base)!=payload["base_state_identity"]:
        raise ValueError("original state differs from overlay parent")
    q=reverse.generator_q(base.jpi_p.numpy(),base.rpp.numpy(),base.storage_base.numpy())
    if reverse.array_hash(q)!=payload["Q_sha256"] or receipt["Q_sha256"]!=payload["Q_sha256"]:
        raise ValueError("loaded constants differ from analytic input")
    before=state_identity(base)
    candidate=attach_overlay(base,payload)
    if state_identity(base)!=before:raise ValueError("original object changed")
    return candidate


def create_overlay(candidate_id,base,protocol,folder):
    label=CANDIDATES[candidate_id]
    parent=Path(previous.model_spec(label)["export"])
    if sha(parent)!=PARENT_SHA[label]:raise ValueError("fixed original model changed")
    q=reverse.generator_q(base.jpi_p.numpy(),base.rpp.numpy(),base.storage_base.numpy())
    math_path=PRIVATE/("math_"+reverse.array_hash(q)+".json")
    if math_path.exists():
        receipt=read(math_path)
        if receipt["Q_sha256"]!=reverse.array_hash(q) or receipt["analytic_implementation_sha256"]!=sha(Path(__file__)):raise ValueError("existing math receipt differs")
        reference=np.asarray(receipt["B_ref"],dtype=np.float64)
    else:
        reference,receipt=analytic_reference(q)
        write_once(math_path,receipt)
    if not receipt["qualified"] or reverse.array_hash(reference)!=receipt["B_ref_sha256"]:raise ValueError("mathematical entry gate failed")
    copied=reverse.replace_copy(base,reference)
    payload={"format":FORMAT,"candidate_id":candidate_id,"parent_label":label,"parent":ident(parent),"parent_format":read(parent)["format"],"source_closure":source_closure(),"protocol":ident(protocol),
        "patch_key":"p_backward","original_B_sha256":reverse.tensor_hash(base.p_backward),"Q_sha256":reverse.array_hash(q),"B_ref_sha256":reverse.array_hash(reference),"B_ref_shape":list(reference.shape),"B_ref":reference.tolist(),
        "base_state_identity":state_identity(base),"candidate_state_identity":state_identity(copied),"math_receipt":ident(math_path),"analytic_implementation_version":FORMAT}
    path=folder/"candidate_overlay.json"
    write_once(path,payload)
    candidate=load_overlay(path)
    # Export is explicitly the new format; round-trip never uses old format.
    candidate.export(folder/"candidate_roundtrip.json")
    reloaded=load_overlay(folder/"candidate_roundtrip.json")
    if state_identity(candidate)!=state_identity(reloaded) or read(path)!=read(folder/"candidate_roundtrip.json"):
        raise ValueError("overlay load/export/reload is not bitwise stable")
    return reloaded,{"overlay":ident(path),"parent":ident(parent),"math_receipt":ident(math_path),"roundtrip_bitwise_equal":True,"only_buffer":assert_only_buffer(base,reloaded)}


def causal_entry(model,histories):
    import p_port_diagnostic_20261031 as filters
    records={}
    legal=np.asarray(histories,dtype=np.float64).copy();legal[:,:-1,20:30]=0.
    changed=legal.copy();changed[:,:-1,20:30]=np.arange(100).reshape(10,10)+13.
    all_ok=True
    for dt in (.0025,.00125):
        original,trace=reverse.capture_initialize(model,legal,dt,trace=True)
        with torch.no_grad():
            p_ref=np.stack([filters.linear_filter(h[::-1,10:20],h[-1,20:30],model.jpi_p.numpy(),(model.rpp*model.storage_base).numpy(),-.01)[::-1] for h in legal])
            model.clear_cache();a=model.initialize(torch.as_tensor(changed),history_dt=dt).numpy()
            model.begin_graph_cache()
            try:b=model.initialize(torch.as_tensor(legal),history_dt=dt).numpy()
            finally:model.end_graph_cache()
        flags={"actual_candidate_buffer_active":trace["status"]=="active_buffer","reverse_history_original_1e12":bool(np.allclose(original["p_history"],p_ref,atol=1e-12,rtol=1e-12)),
            "earlier_p_ignored_bitwise":bool(np.array_equal(a,original["initial"])),"empty_hot_cache_1e12":bool(np.allclose(b,original["initial"],atol=1e-12,rtol=1e-12)),"terminal_observation_pinned":bool(np.array_equal(original["initial"][:,:30],legal[:,-1,:30]))}
        records[str(dt)]={"gates":flags,"trace":trace,"history_maxabs_vs_independent":float(np.max(np.abs(original["p_history"]-p_ref)))}
        all_ok &= all(flags.values())
    return {"passed":bool(all_ok),"history_steps":records}


def scalar_score_gate(new,old,u_new,u_old):
    values=(new,old,u_new,u_old)
    if not all(isinstance(x,(int,float)) and not isinstance(x,bool) and np.isfinite(x) for x in values):
        return {"passed":False,"reason":"unresolved_scalar","delta":None,"allowance":None}
    return {"passed":bool(abs(new-old)<=u_new+u_old+1e-12),"delta":abs(new-old),"allowance":u_new+u_old+1e-12,"new":new,"old":old,"u_new":u_new,"u_old":u_old}


def category_sets(rows):
    result={key:set() for key in ("detected_unstable","missed_unstable","false_los","numerical_unresolved","latency_uncovered","window_no_los_unsettled")}
    for r in rows:
        eid=r["event_id"]
        if not r["numerically_admissible"]:result["numerical_unresolved"].add(eid)
        else:
            if r["label"]=="unstable":result["detected_unstable" if r["predicted_los"] else "missed_unstable"].add(eid)
            elif r["predicted_los"]:result["false_los"].add(eid)
        for category in ("latency_uncovered","window_no_los_unsettled"):
            if r["category"]==category:result[category].add(eid)
    return {key:sorted(ids) for key,ids in result.items()}


def fixed_timing_gate(old,new):
    from checkpoint_selection_decision_20261030 import ready,errors
    anchor=sorted(eid for eid,row in old.items() if ready(row))
    missing=[eid for eid in anchor if not ready(new.get(eid,{}))]
    truth_mismatch=[eid for eid in anchor if eid in new and new[eid]["truth"]["time_s"]!=old[eid]["truth"]["time_s"]]
    if not anchor or missing or truth_mismatch:return {"passed":False,"T_base":anchor,"missing_ids":missing,"truth_mismatch_ids":truth_mismatch,"reason":"empty_or_incomplete_fixed_T"}
    a,b=errors({"timing_actual15":old},anchor),errors({"timing_actual15":new},anchor)
    per={eid:abs(new[eid]["fine"]["time_s"]-old[eid]["fine"]["time_s"])<=.01+abs(new[eid]["fine"]["time_s"]-new[eid]["coarse"]["time_s"])+abs(old[eid]["fine"]["time_s"]-old[eid]["coarse"]["time_s"]) for eid in anchor}
    return {"passed":bool(b["E"]<=a["E"]+.01+a["u"]+b["u"] and all(per.values())),"T_base":anchor,"old":a,"new":b,"event_fine_time_gates":per,"missing_ids":[],"truth_mismatch_ids":[]}


def cross_interval(a,b,event,mass,pscale,stride,n,*,classification_only=False):
    """Original COI/common arithmetic, on the full registered same-grid interval."""
    from evaluate_structured_rom import _pair_row,_scored_numerics,SCALES
    full_a,full_b=a[:stride*n+1],b[:stride*n+1]
    pair=_pair_row(full_a,full_b[::stride],event["x"][:11],event,event["classification_meta"],mass,stride=stride)
    # The original helper supplies the 1/2/5/8/prefix score rows.
    scored=_scored_numerics(a,b,event,mass,stride=stride,coarse_stride=stride)
    scored=[row for row in scored if not row["available"] or (min(800,len(event["x"])-11) if row["horizon"]=="prefix" else int(row["horizon"])*100)<=n]
    endpoint=pair["comparison_last_step"] if classification_only else n
    full_a,full_b=full_a[:stride*endpoint+1],full_b[:stride*endpoint+1]
    finite=bool(np.isfinite(full_a).all() and np.isfinite(full_b).all())
    if finite:
        delta=full_a[::stride]-full_b[::stride];de=delta[:,:10];we=delta[:,10:20]
        dc=de-(de*mass).sum(1,keepdims=True)/mass.sum();wc=we-(we*mass).sum(1,keepdims=True)/mass.sum()
        coi=float(np.sqrt(.5*np.mean((dc/SCALES[0])**2+(wc/SCALES[1])**2)))
        common=float(np.sqrt(np.mean(((we*mass).sum(1)/mass.sum()/SCALES[1])**2)))
        power=float(np.sqrt(np.mean((delta[1:,20:30]/pscale)**2))) if len(delta)>1 else None
    else:coi=common=power=None
    from fixed_edge_pair_20261019 import scientific_qualification
    good=finite and scientific_qualification(pair,scored) and coi<=.001 and common<=.001 and power is not None and power<=.001
    return {"qualified":bool(good),"finite_full_interval":finite,"coi_normalized_rmse":coi,"common_normalized_rmse":common,"p_normalized_rms":power,"classification_endpoint_pair":pair,"original_scored_rows":scored,"output_steps":n}


def prediction_blocks(files):
    for entry in files:
        if sha(entry["path"])!=entry["sha256"]:raise ValueError("frozen prediction hash differs")
        with np.load(entry["path"],allow_pickle=False) as data:
            ids=[str(x) for x in data["event_ids"]]
            h,f,c=data["histories"],data["fine"],data["coarse"]
            if h.shape!=(len(ids),11,30) or f.shape!=(len(ids),3001,40) or c.shape!=(len(ids),1501,40):raise ValueError("paired prediction shapes changed")
            yield ids,h,f,c


def path_score_rows(model,event,path):
    from train_structured_rom import _point_error
    from evaluate_l1_latent import _common_frequency_row
    result={}
    for h,n in [(str(h),h*100) for h in (1,2,5,8)]+[("actual15",min(1500,len(event["x"])-11))]:
        if n<=0 or len(event["x"])-11<n or not np.isfinite(path[:n+1]).all():
            result[h]=None;continue
        pred,truth=path[1:n+1,:30],event["x"][11:11+n,:30]
        with torch.no_grad():point=float(_point_error(torch.as_tensor(pred),torch.as_tensor(truth),model).mean())
        row={"point_error":point}
        if h=="8":
            frequency=_common_frequency_row(path,event["x"][10:],model.mass.numpy(),n,h)
            power=(pred[:,20:30]-truth[:,20:30])/model.p_scale.numpy()
            row.update(common_sse=frequency["sse"],common_n=frequency["n"],p_sse=float(np.sum(power**2)),p_n=int(power.size))
        result[h]=row
    return result


def aggregate_scores(rows):
    from train_local_memory_0098 import risk
    out={}
    for h in ("1","2","5","8","actual15"):
        group=[r for r in rows if r["label"]==("unstable" if h=="actual15" else "stable")]
        out[h]=float(risk(torch.tensor([r["scores"][h]["point_error"] for r in group],dtype=torch.float64))) if group and all(r["scores"].get(h) is not None for r in group) else None
    stable=[r["scores"]["8"] for r in rows if r["label"]=="stable"]
    for label,sse,n in (("common_frequency_rmse_8s","common_sse","common_n"),("normalized_p_rmse_8s","p_sse","p_n")):
        out[label]=float(np.sqrt(sum(r[sse] for r in stable)/sum(r[n] for r in stable))) if stable and all(r is not None for r in stable) else None
    return out


def baseline_description(model,summary,detail,events,*,strict_baseline=True):
    from evaluate_structured_rom import _classification
    lookup={e["id"]:e for e in events};fine_rows=[];coarse_rows=[];seen=[]
    classification={scope:{"fine":[],"coarse":[]} for scope in ("8s","actual15")}
    qmap={scope:{r["event_id"]:r["numerically_admissible"] for r in detail["classification"][scope]["rows"]} for scope in classification}
    for ids,h,f,c in prediction_blocks(summary["prediction_files"]):
        for eid,history,ff,cc in zip(ids,h,f,c):
            event=lookup[eid]
            if not np.array_equal(history,event["x"][:11]):raise ValueError("baseline history/cache mismatch")
            seen.append(eid)
            fine_rows.append({"event_id":eid,"label":event["label"],"scores":path_score_rows(model,event,ff[::2])})
            coarse_rows.append({"event_id":eid,"label":event["label"],"scores":path_score_rows(model,event,cc)})
            for scope,n in (("8s",800),("actual15",min(1500,len(event["x"])-11))):
                for name,path in (("fine",ff[::2][:n+1]),("coarse",cc[:n+1])):
                    cl=_classification(path,history,event["classification_meta"],model.mass.numpy())
                    classification[scope][name].append({"event_id":eid,"label":event["label"],"numerically_admissible":qmap[scope][eid],**cl})
    if len(seen)!=297 or len(set(seen))!=297 or set(seen)!=set(lookup):raise ValueError("baseline prediction cohort differs")
    scores={"fine":aggregate_scores(fine_rows),"coarse":aggregate_scores(coarse_rows)}
    p=summary["decision_profile"]
    expected={"5":p["horizon_risks"]["5"]["value"],"8":p["horizon_risks"]["8"]["value"],"actual15":p["unstable_actual15_risk"],"common_frequency_rmse_8s":p["common_frequency_rmse_8s"],"normalized_p_rmse_8s":p["normalized_p_rmse_8s"]}
    reproduction={}
    for key,value in expected.items():
        raw=scores["fine"][key]
        passed=value is not None and raw is not None and abs(value-raw)<=1e-12
        reproduction[key]={"passed":passed,"official":value,"recomputed_finite_path_diagnostic":raw,"status":"reproduced" if passed else "unresolved" if value is None or raw is None else "mismatch"}
        if strict_baseline and not passed:raise ValueError("baseline original metric reproduction differs")
    if not strict_baseline:
        # A legitimate failed numerical qualification must finish with null
        # official scores, never a finite-only surrogate or an execution error.
        for key,value in expected.items():scores["fine"][key]=value
        for h in ("1","2","5","8"):
            official=summary.get("official_horizon_risks",{}).get(h,p.get("horizon_risks",{}).get(h))
            if official is not None:
                scores["fine"][h]=official.get("value")
                scores["coarse"][h]=official.get("coarse_value")
            elif h in ("1","2") and not p.get("all_8s_rows_qualified"):
                scores["fine"][h]=scores["coarse"][h]=None
        scores["coarse"]["actual15"]=p.get("unstable_actual15_coarse_risk")
        for key in ("common_frequency_rmse_8s","normalized_p_rmse_8s"):
            if scores["fine"][key] is None:scores["coarse"][key]=None
    return {"scores":scores,"metric_reproduction":reproduction,"event_count":len(seen),"stable_denominator":sum(e["label"]=="stable" for e in events),"unstable_denominator":sum(e["label"]=="unstable" for e in events),
        "classification":classification,"sets":{scope:category_sets(cell["fine"]) for scope,cell in classification.items()},"timing_actual15":p["timing_actual15"],"profile":p,"prediction_files":summary["prediction_files"],"original_inference_cost":summary["original_inference_cost"]}


def verify_parent_archive():
    path=PARENT/"archival_verified.json"
    proof=read(path)
    if proof.get("ordinary_push_verified") is not True or proof.get("local_commit")!=PARENT_COMMIT or proof.get("remote_commit")!=PARENT_COMMIT:
        raise ValueError("reverse probe archival_verified.json does not verify matching local/remote commits at the frozen parent")
    return path,proof


def register_prepare(protocol,design):
    if PRIVATE.exists():raise FileExistsError("prepare directory exists; reconcile before any replay; keep wrapper logs outside")
    PRIVATE.mkdir(parents=True)
    started=time.perf_counter()
    try:
        archive,proof=verify_parent_archive()
        parent_complete=read(PARENT/"complete.json")
        if parent_complete.get("status")!="COMPLETED":raise ValueError("reverse probe not complete")
        frozen=read(previous.PRIVATE/"manifest.json");frozen_sha=sha(previous.PRIVATE/"manifest.json")
        summaries={};details={};models={};bindings={}
        for label in ("A96","step000016","R_N"):
            folder=previous.PRIVATE/label
            summary=previous.verify_complete(folder,frozen_sha)
            if summary["model"]["sha256"]!=(PARENT_SHA[label] if label in PARENT_SHA else frozen["models"][label]["sha256"]):raise ValueError("baseline fixed model differs")
            summaries[label]=summary;details[label]=read(ROOT/summary["per_event"]["path"])
            import state_gated_memory_study_20261028 as sgm
            if not sgm.valid(summary["decision_profile"]):raise ValueError("original baseline numerical qualification unresolved")
            if label!="R_N":models[label]=strict_base_loader(label,ROOT/summary["model"]["path"])
            bindings[label]={"summary":ident(folder/"summary.json"),"complete":ident(folder/"complete.json"),"per_event":summary["per_event"],"model":summary["model"],"predictions":summary["prediction_files"]}
            if label!="R_N":
                original=Path(summary["prediction_files"][0]["path"]).parent
                proof_old=read(original/"complete.json")
                if proof_old["status"]!="COMPLETED" or proof_old["event_count"]!=297 or proof_old["evidence_sha256"]!=sha(original/"evidence.json"):
                    raise ValueError("original baseline evidence completion receipt differs")
                bindings[label]["original_evidence"]=ident(original/"evidence.json")
                bindings[label]["original_evidence_complete"]=ident(original/"complete.json")
        cache=ROOT/frozen["data"]["fit_cache"]["path"]
        if sha(cache)!=frozen["data"]["fit_cache"]["sha256"]:raise ValueError("frozen fit cache changed")
        anchor_path=previous.PRIVATE/"anchors.json";anchors=read(anchor_path)
        if anchors["manifest_sha256"]!=frozen_sha or anchors["reference_summary_hashes"]!={key:bindings[key]["summary"]["sha256"] for key in ("R_N","A96")}:raise ValueError("original science anchors changed")
        manifest={"format":FORMAT,"protocol":ident(protocol),"design":ident(design),"source":source_closure(),"reference_source_closure":__import__("state_gated_memory_study_20261028").code_identity(),"parent_complete":ident(PARENT/"complete.json"),"parent_archival_verified":ident(archive),"previous_manifest":ident(previous.PRIVATE/"manifest.json"),"data":frozen["data"],"baselines":bindings,"science_anchors":ident(anchor_path),"normal_full_paired_limit":2,"repair_full_paired_authorized":False,"optimizer_calls":0,"parameter_estimates":0,"new_fom":0,"val_read":False,"test_read":False}
        write_once(PRIVATE/"manifest.json",manifest)
        with np.load(cache,allow_pickle=False) as data:
            metadata=json.loads(str(data["metadata_json"]))
            events=[{**e,"x":data[f"event_{i:04d}_x"].copy()} for i,e in enumerate(metadata)]
        if len(events)!=297 or {e["id"]:e["label"] for e in events}!=frozen["data"]["event_labels"]:raise ValueError("fit297 cohort changed")
        baselines={}
        for label,model in models.items():
            baselines[label]=baseline_description(model,summaries[label],details[label],events)
        baselines["R_N"]={"profile":summaries["R_N"]["decision_profile"]}
        write_once(PRIVATE/"baselines.json",baselines)
        write_once(PRIVATE/"science_anchors.json",anchors)
        math_times={};entry={}
        for candidate_id,label in CANDIDATES.items():
            stage=time.perf_counter();folder=PRIVATE/candidate_id
            try:
                candidate,identity=create_overlay(candidate_id,models[label],protocol,folder)
                math_times[candidate_id]=time.perf_counter()-stage
                entry[candidate_id]=causal_entry(candidate,np.stack([e["x"][:11] for e in sorted(events,key=lambda e:e["id"])]))
                payload={"candidate_identity":identity,"causal_entry":entry[candidate_id],"buffer_and_load_seconds":math_times[candidate_id]}
            except Exception as exc:
                entry[candidate_id]={"passed":False,"status":"ENTRY_BLOCKED","error":repr(exc)}
                payload={"candidate_identity":None,"causal_entry":entry[candidate_id],"buffer_and_load_seconds":time.perf_counter()-stage}
            write_once(folder/"entry.json",payload)
        write_once(PRIVATE/"prepare_complete.json",{"status":"COMPLETED","manifest_sha256":sha(PRIVATE/"manifest.json"),"baselines":ident(PRIVATE/"baselines.json"),"science_anchors":ident(PRIVATE/"science_anchors.json"),"entries":{key:ident(PRIVATE/key/"entry.json") for key in CANDIDATES},"wall_seconds":time.perf_counter()-started,"new_ROM_rollouts":0,"optimizer_calls":0})
        return entry
    except BaseException as exc:
        write_once(PRIVATE/"prepare_failure.json",{"error":repr(exc),"traceback":traceback.format_exc(),"wall_seconds":time.perf_counter()-started,"new_ROM_rollouts":0})
        raise


def verify_prepared():
    complete=read(PRIVATE/"prepare_complete.json");manifest=read(PRIVATE/"manifest.json")
    import state_gated_memory_study_20261028 as sgm
    if complete["status"]!="COMPLETED" or complete["manifest_sha256"]!=sha(PRIVATE/"manifest.json") or manifest["source"]!=source_closure() or manifest["reference_source_closure"]!=sgm.code_identity():raise ValueError("prepared source identity differs")
    for item in [complete["baselines"],complete["science_anchors"],*complete["entries"].values(),manifest["protocol"],manifest["design"],manifest["data"]["fit_cache"],manifest["data"]["prepare"]]:
        if sha(ROOT/item["path"])!=item["sha256"]:raise ValueError("prepared artifact changed")
    return manifest


def science_entry(profile,detail,baselines,anchors):
    from memory_transport_study_20261029 import efficiency_target
    from checkpoint_selection_decision_20261030 import detections,ready,errors
    target=efficiency_target(profile,baselines["A96"]["profile"],baselines["R_N"]["profile"])
    protected={scope:set(ids)<=detections(detail,scope) for scope,ids in anchors["detection_union"].items()}
    ids=anchors["T_anchor"];time=profile["timing_actual15"]
    coverage=bool(ids) and all(ready(time.get(eid,{})) for eid in ids)
    truth=coverage and all(time[eid]["truth"]["time_s"]==anchors["truth_times"][eid] for eid in ids)
    gates={}
    for ref in ("A96","R_N"):
        reference=baselines[ref]["profile"]
        old=errors(reference,ids);new=errors(profile,ids)
        gates[ref]=bool(coverage and truth and new["E"]<=old["E"]+.01+new["u"]+old["u"])
    return {"science_entry_eligible":target.get("passed") is True and all(protected.values()) and all(gates.values()),"original_efficiency_target":target,"protected_detection_union":protected,"T_anchor":ids,"T_anchor_coverage":coverage,"T_anchor_truth":truth,"T_anchor_timing_gates":gates,"not_the_implementation_T_base":True}


def compare_candidate(model,events,old,new):
    lookup={e["id"]:e for e in events};rows=[];seen=[]
    old_files,new_files=old["prediction_files"],new["prediction_files"]
    if len(old_files)!=len(new_files):raise ValueError("paired block layout differs")
    for first,second in zip(prediction_blocks(old_files),prediction_blocks(new_files)):
        ids,h,f,c=first;newids,hh,ff,cc=second
        if ids!=newids or not np.array_equal(h,hh):raise ValueError("same-event paired prediction alignment differs")
        for eid,oldf,oldc,newf,newc in zip(ids,f,c,ff,cc):
            event=lookup[eid];seen.append(eid);intervals={}
            horizons=[(str(h),h*100) for h in (1,2,5,8)] if event["label"]=="stable" else [("actual15",min(1500,len(event["x"])-11))]
            for grid,a,b,stride in (("fine",oldf,newf,2),("coarse",oldc,newc,1)):
                for scope,n in horizons:
                    intervals[f"{grid}_{scope}"]=cross_interval(a,b,event,model.mass.numpy(),model.p_scale.numpy(),stride,n)
                intervals[f"{grid}_classification8"]=cross_interval(a,b,event,model.mass.numpy(),model.p_scale.numpy(),stride,800,classification_only=True)
            rows.append({"event_id":eid,"label":event["label"],"qualified":all(cell["qualified"] for cell in intervals.values()),"intervals":intervals})
    if len(seen)!=297 or len(set(seen))!=297 or set(seen)!=set(lookup):raise ValueError("new-old comparison lost original events")
    metrics={}
    for key in ("5","8","actual15","common_frequency_rmse_8s","normalized_p_rmse_8s"):
        of,oc=old["scores"]["fine"][key],old["scores"]["coarse"][key]
        nf,nc=new["scores"]["fine"][key],new["scores"]["coarse"][key]
        un=abs(nf-nc) if nf is not None and nc is not None else None
        uo=abs(of-oc) if of is not None and oc is not None else None
        metrics[key]=scalar_score_gate(nf,of,un,uo)
    category={};changes={}
    for scope in ("8s","actual15"):
        category[scope]={"frozen_sets_equal":old["sets"][scope]==new["sets"][scope]}
        changes[scope]={}
        for grid in ("fine","coarse"):
            a={r["event_id"]:r for r in old["classification"][scope][grid]};b={r["event_id"]:r for r in new["classification"][scope][grid]}
            changes[scope][grid]={"category_changed_ids":[eid for eid in sorted(a) if a[eid]["category"]!=b[eid]["category"]],"window_settled_changed_ids":[eid for eid in sorted(a) if a[eid].get("window_settled")!=b[eid].get("window_settled")]}
            changes[scope][grid]["category_identity"]=not changes[scope][grid]["category_changed_ids"]
            changes[scope][grid]["window_settled_identity"]=not changes[scope][grid]["window_settled_changed_ids"]
    timing=fixed_timing_gate(old["timing_actual15"],new["timing_actual15"])
    passed=all(r["qualified"] for r in rows) and all(v["passed"] for v in metrics.values()) and all(all(cell.values()) for cell in category.values()) and timing["passed"]
    return {"passed":bool(passed),"record_count":len(rows),"cross_grid_all_qualified":all(r["qualified"] for r in rows),"score_gates":metrics,"classification_gates":category,"classification_changes":changes,"T_base_timing":timing,"per_event":rows}


def checked_candidate_complete(folder):
    receipt=read(folder/"complete.json")
    if receipt["status"]!="COMPLETED" or receipt["manifest_sha256"]!=sha(PRIVATE/"manifest.json"):raise ValueError("candidate completion identity differs")
    for item in receipt["artifacts"]:
        if sha(ROOT/item["path"])!=item["sha256"]:raise ValueError("candidate completed artifact changed")
    return read(folder/"qualification.json")


def evaluate(candidate_id):
    manifest=verify_prepared();folder=PRIVATE/candidate_id
    if (folder/"complete.json").exists():return checked_candidate_complete(folder)
    if (folder/"started.json").exists():raise RuntimeError("candidate has an existing attempt; reconcile first, no automatic rerun or repair authorization")
    entry=read(folder/"entry.json")
    if not entry["causal_entry"]["passed"]:
        result={"status":"ENTRY_BLOCKED","new_full_paired_evaluations":0,"entry":entry,"implementation_qualified":False,"candidate_id":candidate_id}
        write_once(folder/"qualification.json",result)
        write_once(folder/"complete.json",{"status":"COMPLETED","manifest_sha256":sha(PRIVATE/"manifest.json"),"artifacts":[ident(folder/"qualification.json"),ident(folder/"entry.json")]})
        return result
    started=time.perf_counter()
    write_once(folder/"started.json",{"manifest_sha256":sha(PRIVATE/"manifest.json"),"candidate_id":candidate_id,"started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"normal_attempt":1,"optimizer_calls":0,"parameter_estimates":0})
    try:
        import state_gated_memory_study_20261028 as sgm
        cache=ROOT/manifest["data"]["fit_cache"]["path"]
        with np.load(cache,allow_pickle=False) as data:
            metadata=json.loads(str(data["metadata_json"]));events=[{**e,"x":data[f"event_{i:04d}_x"].copy()} for i,e in enumerate(metadata)]
        model=load_overlay(folder/"candidate_overlay.json")
        if sha(folder/"candidate_overlay.json")!=entry["candidate_identity"]["overlay"]["sha256"]:raise ValueError("candidate composite overlay identity changed")
        model_id=state_identity(model)
        if state_identity(copy.deepcopy(model))!=model_id:raise ValueError("deepcopy loses analytic patch")
        prepared=read(ROOT/manifest["data"]["prepare"]["path"])
        print(json.dumps({"candidate":candidate_id,"status":"FULL_PAIRED_STARTED","optimizer_calls":0}),flush=True)
        stage=time.perf_counter()
        # Exactly one new full evaluation in this candidate's normal attempt.
        with torch.no_grad():evidence=sgm.evaluate(model,events,folder/"paired",prepared)
        evaluation_seconds=time.perf_counter()-stage
        if state_identity(model)!=model_id:raise ValueError("qualification mutated candidate")
        profile=evidence["decision_profile"]
        detail={"classification":{scope:previous.classification_detail(evidence["extras"],profile,key) for scope,key in (("8s","classification_8s"),("actual15","classification_15s"))}}
        fake_summary={"prediction_files":evidence["prediction_files"],"decision_profile":profile,"official_horizon_risks":evidence.get("profiles",{}).get("fit",{}).get("horizon_risks",{}),"original_inference_cost":{"seconds":evidence["seconds"],"timing":evidence["timing"]}}
        new=baseline_description(model,fake_summary,detail,events,strict_baseline=False)
        baselines=read(PRIVATE/"baselines.json");old=baselines[CANDIDATES[candidate_id]]
        comparison=compare_candidate(model,events,old,new)
        scientific=science_entry(profile,detail,baselines,read(PRIVATE/"science_anchors.json"))
        result={"format":FORMAT,"candidate_id":candidate_id,"status":"COMPLETED","manifest_sha256":sha(PRIVATE/"manifest.json"),"candidate_identity":entry["candidate_identity"],
            "implementation_qualified":bool(entry["causal_entry"]["passed"] and sgm.valid(profile) and comparison["passed"]),"original_numerical_qualification":bool(sgm.valid(profile)),"causal_entry":entry["causal_entry"],"paired_behavior":{key:value for key,value in comparison.items() if key!="per_event"},"science_entry":scientific,
            "profile":profile,"candidate_metric_reproduction":new["metric_reproduction"],"denominators":{"event_count":new["event_count"],"stable":new["stable_denominator"],"unstable":new["unstable_denominator"]},"stable_1_2_5_8_scores":new["scores"],"original_inference_cost_reused":old["original_inference_cost"],"new_cost":{"full_paired_seconds":evaluation_seconds,"paired_timing":evidence["timing"],"buffer_and_load_seconds":entry["buffer_and_load_seconds"],"wall_seconds_before_publication":time.perf_counter()-started},
            "same_precision_cost_comparison":"not_compared; old cached timings are separate; no speed claim","full_paired_evaluations":1,"optimizer_calls":0,"parameter_estimates":0,"new_fom":0,"model_adopted":False,"old_M3_unlocked":False,"p_unit_note":"frozen numerical p coordinate; physical helper MW/power labels are not independent unit certification"}
        write_once(folder/"comparison_per_event.json",comparison["per_event"])
        write_once(folder/"classification_and_times.json",{"classification":new["classification"],"sets":new["sets"],"timing_actual15":new["timing_actual15"]})
        write_once(folder/"qualification.json",result)
        artifacts=[ident(folder/name) for name in ("candidate_overlay.json","candidate_roundtrip.json","entry.json","comparison_per_event.json","classification_and_times.json","qualification.json")]
        artifacts.append(entry["candidate_identity"]["math_receipt"])
        artifacts += [{"path":Path(e["path"]).resolve().relative_to(ROOT.resolve()).as_posix(),"sha256":e["sha256"]} for e in evidence["prediction_files"]]
        artifacts += [ident(folder/"paired"/name) for name in ("evidence.json","state_gated_profile.json","complete.json")]
        write_once(folder/"complete.json",{"status":"COMPLETED","manifest_sha256":result["manifest_sha256"],"artifacts":artifacts,"actual_total_wall_seconds":time.perf_counter()-started})
        print(json.dumps({"candidate":candidate_id,"status":"COMPLETED","implementation_qualified":result["implementation_qualified"],"science_entry_eligible":scientific["science_entry_eligible"]}),flush=True)
        return result
    except BaseException as exc:
        write_once(folder/"failure.json",{"status":"FAILED","error":repr(exc),"traceback":traceback.format_exc(),"wall_seconds":time.perf_counter()-started,"full_prediction_attempt_may_have_executed":True,"automatic_retry_allowed":False,"optimizer_calls":0})
        raise


def aggregate_public(result):
    if result["status"]=="ENTRY_BLOCKED":return {"candidate_id":result["candidate_id"],"status":"ENTRY_BLOCKED","implementation_qualified":False,"new_full_paired_evaluations":0,"model_adopted":False}
    comparison=result["paired_behavior"];science=result["science_entry"]
    timing=comparison["T_base_timing"]
    return {"candidate_id":result["candidate_id"],"status":result["status"],"implementation_qualified":result["implementation_qualified"],"original_numerical_qualification":result["original_numerical_qualification"],"causal_entry_passed":result["causal_entry"]["passed"],"score_gates":comparison["score_gates"],"classification_gates":comparison["classification_gates"],"cross_grid_all_qualified":comparison["cross_grid_all_qualified"],
        "T_base":{"passed":timing["passed"],"count":len(timing["T_base"]),"sha256":hashlib.sha256(json.dumps(timing["T_base"]).encode()).hexdigest(),"missing_count":len(timing.get("missing_ids",[])),"truth_mismatch_count":len(timing.get("truth_mismatch_ids",[])),"old":timing.get("old"),"new":timing.get("new")},
        "science_entry":{"science_entry_eligible":science["science_entry_eligible"],"protected_detection_union":science["protected_detection_union"],"T_anchor_count":len(science["T_anchor"]),"T_anchor_coverage":science["T_anchor_coverage"],"T_anchor_timing_gates":science["T_anchor_timing_gates"]},
        "denominators":result["denominators"],"stable_1_2_5_8_scores":result["stable_1_2_5_8_scores"],"new_cost":result["new_cost"],"optimizer_calls":0,"parameter_estimates":0,"new_fom":0,"model_adopted":False,"old_M3_unlocked":False,"p_unit_note":result["p_unit_note"]}


def summary():
    verify_prepared()
    results={candidate:checked_candidate_complete(PRIVATE/candidate) for candidate in CANDIDATES}
    result={"format":FORMAT,"status":"COMPLETED","candidates":{key:aggregate_public(value) for key,value in results.items()},"both_implementation_qualified":all(r["implementation_qualified"] for r in results.values()),"model_adopted":False,"default_changed":False,"old_M3_unlocked":False,"independent_review_required_for_adoption":True,"manifest_sha256":sha(PRIVATE/"manifest.json")}
    path=PUBLIC/"public_qualification.json"
    if path.exists():
        if read(path)!=safe(result):raise ValueError("immutable public qualification differs")
    else:write_once(path,result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["prepare","evaluate","summary"])
    parser.add_argument("--candidate",choices=list(CANDIDATES))
    parser.add_argument("--protocol",type=Path,default=ROOT/"reports/analytic_p_backward_qualification_20261031_protocol.md")
    parser.add_argument("--design",type=Path,default=ROOT/"reports/analytic_p_backward_qualification_20261031_scientist.md")
    args=parser.parse_args(argv);torch.set_num_threads(1)
    if args.command=="prepare":register_prepare(args.protocol,args.design)
    elif args.command=="evaluate":
        if args.candidate is None:parser.error("evaluate requires --candidate")
        evaluate(args.candidate)
    else:summary()


if __name__=="__main__":
    main()
