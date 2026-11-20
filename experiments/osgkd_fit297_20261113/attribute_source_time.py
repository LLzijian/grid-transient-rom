"""Attribute saved port integration residuals to the nominal source step."""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = "osgkd_fit297_20261113"
PUBLIC = ROOT / "results" / RUN
ADOPTION = PUBLIC / "source_time_adoption.json"
OUTPUT = PUBLIC / "source_time_attribution.json"
INPUTS = [
    f"artifacts/{RUN}/port_audit/recheck_vectors.json.gz",
    f"results/{RUN}/port_audit.json",
    f"results/{RUN}/port_audit_recheck.json",
]
GATE = .002


def checked_bytes(reference):
    payload = (ROOT / reference["path"]).read_bytes()
    if len(payload) != reference["bytes"] or hashlib.sha256(payload).hexdigest() != reference["sha256"]:
        raise ValueError(f"bound file differs: {reference['path']}")
    return payload


def fine_combo(q1, q2):
    return (5*q1-q2)/4


def coarse_combo(q2, q4):
    return (21*q2-5*q4)/16


def analytic_check():
    length, h = Fraction(2,25), Fraction(1,200)
    continuous = length**3/3
    a = length/6
    source, q1, q2, q4 = (continuous+a*(m*h)**2 for m in (1,2,4,8))
    simpson = (4*q1-q2)/3
    assert (source-simpson)/(source-q1) == Fraction(-1,3)
    assert fine_combo(q1,q2) == coarse_combo(q2,q4) == source
    assert source-simpson-(q2-q1)/12 == 0


def identity(actual, expected, scale):
    error = np.abs(actual-expected)
    if not np.isfinite(error).all() or not np.all(error <= 128*np.finfo(np.float64).eps*scale):
        raise ValueError("signed source identity failed")
    return float(error.max())


def rms(vector):
    return float(np.sqrt(np.mean(vector**2)))


def summarize(saved, nested, work):
    def vector(key):
        value = np.asarray(saved[key], dtype=np.float64)
        if value.shape != (10,) or not np.isfinite(value).all():
            raise ValueError(f"invalid saved vector: {key}")
        return value

    impulse, n, q1, q2, s1 = (vector(key) for key in ("impulse","scale","Q01","Q02","S01"))
    if not np.all(n > 0):
        raise ValueError("nonpositive normalization")
    target = fine_combo(q1,q2)
    work["fine_combinations"] += 1
    r_trap, r_simpson, r_source = ((impulse-q)/n for q in (q1,s1,target))
    correction = (q2-q1)/(12*n)
    scale = np.maximum.reduce([np.ones(10), np.abs(impulse), np.abs(q1), np.abs(q2), np.abs(s1)])
    signed_error = identity(r_source, r_simpson-correction, scale/n)
    metrics = dict(r_trap_rms=rms(r_trap), r_simpson_rms=rms(r_simpson), r_source_rms=rms(r_source),
                   correction_rms=rms(correction), simpson_correction_dot_mean=float(np.mean(r_simpson*correction)),
                   signed_identity_max_abs=signed_error)
    if nested:
        q4,s2 = vector("Q04"),vector("S02")
        coarse = coarse_combo(q2,q4)
        work["coarse_combinations"] += 1
        difference = target-coarse
        coarse_error = identity(difference, 5*(q4-5*q2+4*q1)/16,
                                np.maximum(scale,np.abs(q4)))
        metrics.update(r_source_coarse_rms=rms((impulse-coarse)/n),
                       fine_coarse_gap_rms=rms(difference/n), simpson_gap_rms=rms((s2-s1)/n),
                       coarse_identity_max_abs=coarse_error)
    metrics["passed"] = metrics["r_source_rms"] <= GATE
    return metrics


def distributions(rows):
    result = {}
    for window in ("W","W0","W1"):
        metrics = [row["windows"][window] for row in rows]
        result[window] = dict(failed=sum(not item["passed"] for item in metrics), metrics={})
        for key in metrics[0]:
            if key != "passed":
                result[window]["metrics"][key] = dict(zip(("min","p25","median","p75","p95","max"),
                    np.quantile([item[key] for item in metrics], [0,.25,.5,.75,.95,1]).tolist(),strict=True))
    return dict(count=len(rows), failed=sum(not row["passed"] for row in rows), windows=result)


def main():
    release = ADOPTION.read_bytes()
    adoption = json.loads(release)
    work = dict(fine_combinations=0,coarse_combinations=0,raw_source_reads=0,new_integrals=0,model_calls=0)
    if (adoption["run_id"] != RUN or adoption["phase"] != "saved_vector_attribution" or adoption["released"] is not True
            or [item["path"] for item in adoption["inputs"]] != INPUTS
            or adoption["gate"] != {"all_fine_residual_rms_max":GATE,"identities_required":True}
            or adoption["limits"] != dict(work,fine_combinations=891,coarse_combinations=594)):
        raise ValueError("release identity differs")
    wall,cpu = time.perf_counter(),time.process_time()
    receipt = dict(run_id=RUN,status="invalid",adoption_sha256=hashlib.sha256(release).hexdigest(),
                   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   references={k:adoption[k] for k in ("design","review","inputs")}, work=work,
                   gate=GATE,actual_source_step_alignment=None,historical_audits_remain="blocked",
                   threshold_context="post-result correction to target nominal source discretization",
                   training_released=False,rows=[])
    with OUTPUT.open("x",encoding="utf-8") as output:
        try:
            analytic_check()
            checked_bytes(adoption["design"])
            checked_bytes(adoption["review"])
            packed,old_bytes,recheck_bytes = map(checked_bytes,adoption["inputs"])
            vectors = json.loads(gzip.decompress(packed))
            old,recheck = json.loads(old_bytes),json.loads(recheck_bytes)
            ids = old["ordered_ids"]
            old_failed,recheck_failed = set(old["balance_blocked_ids"]),set(recheck["blocked_ids"])
            if (old["status"] != "blocked" or recheck["status"] != "blocked"
                    or len(ids) != len(set(ids)) or len(ids) != 297
                    or ids != recheck["ordered_ids"] or ids != [row["id"] for row in vectors]
                    or ids != [row["id"] for row in recheck["rows"]]
                    or len(old_failed) != 69 or len(recheck_failed) != 191
                    or recheck["private_vectors"] != adoption["inputs"][0]):
                raise ValueError("saved cohort or audit identity differs")
            for item,prior in zip(vectors,recheck["rows"],strict=True):
                event_id = item["id"]
                original_failed = event_id in old_failed
                second_failed = event_id in recheck_failed
                if (prior["original_history_failed"] != original_failed or prior["passed"] == second_failed
                        or set(item["windows"]) != {"W","W0","W1"}):
                    raise ValueError("saved event identity differs")
                windows = {w:summarize(item["windows"][w],w!="W",work) for w in ("W","W0","W1")}
                receipt["rows"].append(dict(id=event_id,first_audit_failed=original_failed,
                    second_audit_failed=second_failed,second_audit_failed_gates=[k for k,v in prior["gates"].items() if not v],
                    windows=windows,passed=all(m["passed"] for m in windows.values())))
            if work != adoption["limits"]:
                raise ValueError("fixed work did not complete")
            rows = receipt["rows"]
            failures = [row["id"] for row in rows if not row["passed"]]
            receipt.update(status="unresolved" if failures else "compatible_at_nominal_source_scale",
                           failed_ids=failures,analytic_check_passed=True,
                           distributions={"all":distributions(rows),
                               "second_audit_failed":distributions([r for r in rows if r["second_audit_failed"]]),
                               "second_audit_passed":distributions([r for r in rows if not r["second_audit_failed"]])},
                           failing_windows=dict(Counter(w for r in rows for w,m in r["windows"].items() if not m["passed"])))
        except Exception as error:
            receipt.update(status="invalid",error_type=type(error).__name__,error=str(error))
            raise
        finally:
            receipt.update(function_wall_seconds=time.perf_counter()-wall,function_cpu_seconds=time.process_time()-cpu,
                           complete_command_wall_seconds=None)
            json.dump(receipt,output,allow_nan=False,separators=(",",":"))
            output.write("\n")
    print(receipt["status"],len(receipt["failed_ids"]),receipt["failing_windows"],work)


if __name__ == "__main__":
    main()
