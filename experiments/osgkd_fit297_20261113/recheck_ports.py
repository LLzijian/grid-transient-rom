"""The one registered OSGKD historical quadrature recheck."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import time

import numpy as np

import audit_ports as original

d, ROOT, RUN = original.d, original.ROOT, original.RUN
RESULT = ROOT/"results"/RUN/"port_audit_recheck.json"
PRIVATE = ROOT/"artifacts"/RUN/"port_audit"/"recheck_vectors.json.gz"
INITIAL_SHA = "42eefe1aa1dc329cd058bd9497cb83bce7e3782c8e0119d5546c284bd9355783"
AMENDMENT = ROOT/"campaigns/osgkd_ieee39/port_audit_amendment.md"
AMENDMENT_SHA = "3ff23f5122b16c9ace888c0b7acdfc195a9cce67a53a6ae1feb1e77e014f135c"
REVIEW = ROOT/"reports"/RUN/"port_amendment_review.md"
REVIEW_SHA = "7168787e20431ebfa6b8f4f7fba1a7c22e036e79a8ee2fb5ed548c5408f8bbee"
EPS = np.finfo(np.float64).eps


def rms(v):
    return float(np.sqrt(np.mean(v*v)))


def quadrature_window(t, omega, tm, te, mass, damping, *, nested, work):
    points = 9 if nested else 11
    original.require(t.shape == (points,) and np.allclose(np.diff(t), .01, atol=1e-9, rtol=0),
                     "recheck window grid differs")
    duration = t[-1]-t[0]
    scale = duration*np.maximum(np.maximum(np.sqrt(np.mean(tm**2,axis=0)),
                                           np.sqrt(np.mean(te**2,axis=0))), .001)
    impulse = mass*(omega[-1]-omega[0])
    force = tm-te-damping*omega

    def integrate(stride, simpson=False):
        values = force[::stride]
        if simpson:
            weights = np.ones(len(values))
            weights[1:-1:2] = 4
            weights[2:-1:2] = 2
            integral = (.01*stride/3)*(weights@values)
        else:
            integral = np.trapezoid(values,t[::stride],axis=0)
        work["history_integral_vectors"] += 1
        return integral

    q1, q2 = integrate(1), integrate(2)
    s1 = integrate(1,simpson=True)
    q4, s2 = (integrate(4),integrate(2,simpson=True)) if nested else (None,None)
    vectors = [impulse,scale,q1,q2,s1] + ([q4,s2] if nested else [])
    original.require(all(np.all(np.isfinite(v)) for v in vectors) and np.all(scale > 0),
                     "nonfinite quadrature vector or scale")
    terms = [np.ones_like(impulse),np.abs(impulse),np.abs(q1),np.abs(q2)]
    if nested:
        terms.append(np.abs(q4))
    tolerance = 128*EPS*np.maximum.reduce(terms)
    identity1 = s1-(4*q1-q2)/3
    identity2 = s2-(4*q2-q4)/3 if nested else None
    original.require(np.all(np.abs(identity1) <= tolerance)
                     and (not nested or np.all(np.abs(identity2) <= tolerance)),
                     "Simpson weight identity mismatch")
    rq1,rq2,rs1 = ((impulse-q)/scale for q in (q1,q2,s1))
    summary = dict(q01_rms=rms(rq1), q02_rms=rms(rq2), original_mesh_difference_rms=rms(rq1-rq2),
                   s01_rms=rms(rs1), s01_identity_max_abs=float(np.max(np.abs(identity1))),
                   s01_absolute_pass=rms(rs1) <= .002)
    evidence = dict(impulse=impulse,scale=scale,Q01=q1,Q02=q2,S01=s1,weight_tolerance=tolerance,
                    s01_identity=identity1)
    if nested:
        d21, d42 = (q2-q1)/scale, (q4-q2)/scale
        defect, limit = rms(d42-4*d21), .5*rms(d42)+.0001
        s_difference = rms((s2-s1)/scale)
        summary.update(q04_rms=rms((impulse-q4)/scale),s02_rms=rms((impulse-s2)/scale),
                       d21_rms=rms(d21),d42_rms=rms(d42),nested_defect_rms=defect,nested_defect_limit=limit,
                       simpson_difference_rms=s_difference,s02_identity_max_abs=float(np.max(np.abs(identity2))),
                       nested_order_pass=defect <= limit,simpson_difference_pass=s_difference <= .002)
        evidence.update(Q04=q4,S02=s2,d21=d21,d42=d42,s02_identity=identity2)
    else:
        summary.update(original_fine_pass=rms(rq1) <= .05,
                       original_mesh_pass=rms(rq1-rq2) <= .02,
                       s01_reduction_pass=rms(rs1) <= .25*rms(rq1)+.0002)
    return summary,{key:vec.tolist() for key,vec in evidence.items()}


def compare_original(now, then):
    if isinstance(then,dict):
        return isinstance(now,dict) and now.keys()==then.keys() and all(compare_original(now[k],v) for k,v in then.items())
    if isinstance(then,list):
        return isinstance(now,list) and len(now)==len(then) and all(compare_original(x,y) for x,y in zip(now,then,strict=True))
    if isinstance(then,float):
        return isinstance(now,(float,int)) and abs(now-then) <= 1e-12*max(1.,abs(then))
    return now==then


def verified_inputs(adoption_path, adoption_sha):
    refs = dict(original_audit=original.reference(original.OUT,INITIAL_SHA),
                amendment=original.reference(AMENDMENT,AMENDMENT_SHA),
                review=original.reference(REVIEW,REVIEW_SHA),
                adoption=original.reference(adoption_path,adoption_sha))
    adoption = json.loads(adoption_path.read_bytes())
    original.require(adoption.get("run")==RUN and adoption.get("released") is True
                     and adoption.get("phase")=="port_recheck", "amendment adoption identity differs")
    refs["source_files"] = {}
    for path in (Path(__file__).resolve(), Path(original.__file__).resolve(), ROOT/"src/kiki/data/osgkd.py"):
        key = path.relative_to(ROOT).as_posix()
        digest = adoption.get("source_hashes",{}).get(key)
        original.require(isinstance(digest,str) and len(digest)==64, f"adoption lacks source hash: {key}")
        refs["source_files"][key] = original.reference(path,digest)
    return refs,json.loads(original.OUT.read_bytes())


def save_private(vectors):
    PRIVATE.parent.mkdir(parents=True,exist_ok=True)
    with PRIVATE.open("xb") as stream:
        stream.write(gzip.compress(d.canonical_json(vectors),mtime=0))
    return original.reference(PRIVATE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adoption",type=Path,required=True)
    parser.add_argument("--adoption-sha",required=True)
    args = parser.parse_args()
    original.require(not RESULT.exists() and not PRIVATE.exists() and not original.BUNDLE.exists(),
                     "the second-pass output already exists")
    refs, previous = verified_inputs(args.adoption.resolve(),args.adoption_sha)
    original.require(previous["status"]=="blocked" and len(previous["balance_blocked_ids"])==69,
                     "original audit state differs")
    RESULT.parent.mkdir(parents=True,exist_ok=True)
    with (RESULT.parent/"port_audit_recheck_claim.json").open("xb") as stream:
        stream.write(d.canonical_json(dict(run=RUN,original_audit=refs["original_audit"],adoption=refs["adoption"])))
    wall,cpu = time.perf_counter(),time.process_time()
    work = dict(events_loaded=0,history_integral_vectors=0)
    receipt = dict(run=RUN,status="invalid",original_audit=refs["original_audit"],references=refs,
                   work=work,rows=[],ode_calls=0,rhs_calls=0,solver_calls=0,training_updates=0)
    vectors = []
    try:
        sources = {p.name:original.reference(p,h) for p,h in original.HASHES.items()}
        sources["common_legacy_tm_offset"] = original.reference(original.BASE)
        sources["frozen_first_audit"] = original.reference(Path(original.__file__))
        for current,old in (("frozen_first_audit","audit"),("common_legacy_tm_offset","common_legacy_tm_offset")):
            original.require(sources[current]==previous["sources"][old],"first source identity differs")
        original.require(all(sources[p.name]==previous["sources"][p.name] for p in original.HASHES),
                         "original input reference differs")
        receipt["sources"] = sources
        white = original.whitelist()
        ids = [row[1] for row in white]
        original.require(ids==previous["ordered_ids"],"original event order differs")
        selected, source_indices = original.selected_metadata(set(ids))
        original.require(source_indices==previous["source_index_refs"],"source index identity differs")
        ref,constants,offset = original.common_reference()
        old_rows = previous["rows"]
        events = []
        with np.load(original.CACHE,allow_pickle=False) as archive:
            desc = json.loads(str(archive["metadata_json"]))
            original.require([(i,x["id"],x["label"]) for i,x in enumerate(desc)]==white,"cached cohort differs")
            keys = {"metadata_json"}|{f"event_{i:04d}_{field}" for i in range(297) for field in ("x","torque")}
            original.require(set(archive.files)==keys,"unexpected payload member")
            for index,meta in enumerate(desc):
                prefix = f"event_{index:04d}_"
                x,torque = archive[prefix+"x"],archive[prefix+"torque"]
                work["events_loaded"] += 1
                event,old = original.inspect_event({prefix+"x":x,prefix+"torque":torque},index,meta,selected,ref,constants,offset)
                original.require(compare_original(old,old_rows[index]),"first audit row does not reproduce")
                events.append(event)
                t = (np.asarray(meta["t"],dtype=np.float64) if "t" in meta else np.arange(len(x))*.01)
                tm,te = x[:,20:]+(offset-ref["Pm"]),torque+(offset-ref["Pm"])
                mass,damping = np.asarray(constants["mass"]),np.asarray(constants["damping"])
                row = dict(id=event["id"],original_history_failed=old["windows"]["history"]["status"]!="passed",
                           original_windows=old["windows"],windows={})
                receipt["rows"].append(row)
                saved = dict(id=event["id"],windows={})
                vectors.append(saved)
                for name,interval in (("W",slice(0,11)),("W0",slice(0,9)),("W1",slice(2,11))):
                    summary,detail = quadrature_window(t[interval],x[interval,10:20],tm[interval],te[interval],
                                                       mass,damping,nested=name!="W",work=work)
                    row["windows"][name],saved["windows"][name] = summary,detail
                w,w0,w1 = (row["windows"][key] for key in ("W","W0","W1"))
                for new,old_key in (("q01_rms","fine_rms"),("q02_rms","coarse_rms"),
                                    ("original_mesh_difference_rms","mesh_difference_rms")):
                    original.require(compare_original(w[new],old["windows"]["history"][old_key]),
                                     "direct historical quadrature differs from original")
                row["gates"] = dict(original_fine=w["original_fine_pass"],original_future=old["windows"]["early_future"]["status"] in ("passed","not_applicable"),
                                    W_absolute=w["s01_absolute_pass"],W_reduction=w["s01_reduction_pass"],
                                    **{f"{name}_{key}":summary[key] for name,summary in (("W0",w0),("W1",w1))
                                       for key in ("s01_absolute_pass","nested_order_pass","simpson_difference_pass")})
                row["passed"] = all(row["gates"].values())
        fit,check = d.split_events(events)
        scales,counts = d.calibrate(fit)
        constants.update({k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in scales.items()})
        constants["constants_id"] = d.sha_bytes(d.canonical_json(constants))
        receipt.update(ordered_ids=ids,fit_ids=[e["id"] for e in fit],check_ids=[e["id"] for e in check],
                       constants_id=constants["constants_id"],group_order=d.group_order(events),
                       calibration_counts=counts)
        original.require(all(receipt[key]==previous[key] for key in ("ordered_ids","fit_ids","check_ids","group_order","calibration_counts","constants_id"))
                         and constants==previous["constants"] and work["history_integral_vectors"]==3861,
                         "frozen calibration or total work differs")
        blocked = [row["id"] for row in receipt["rows"] if not row["passed"]]
        receipt.update(status="blocked" if blocked else "passed",blocked_ids=blocked,
                       failures_by_gate=dict(Counter(key for row in receipt["rows"]
                                                      for key,value in row["gates"].items() if not value)))
        if not blocked:
            receipt["bundle"] = original.export(events,constants)
    except Exception as error:
        receipt.update(status="invalid",error_type=type(error).__name__,error=str(error))
        raise
    finally:
        receipt.update(function_wall_seconds=time.perf_counter()-wall,function_cpu_seconds=time.process_time()-cpu,
                       complete_command_wall_seconds=None)
        try:
            if vectors:
                receipt["private_vectors"] = save_private(vectors)
        except Exception as error:
            receipt.update(status="invalid",evidence_error_type=type(error).__name__)
            raise
        finally:
            with RESULT.open("xb") as stream:
                stream.write(d.canonical_json(receipt))
    print(json.dumps({key:receipt[key] for key in ("status","blocked_ids","failures_by_gate","work")},ensure_ascii=False))


if __name__ == "__main__":
    main()
