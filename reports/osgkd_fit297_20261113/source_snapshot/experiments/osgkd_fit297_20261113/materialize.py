"""One released materialization of frozen observations; no source diagnostics."""
import json
from pathlib import Path
import time

import numpy as np

import audit_ports as audit

d, ROOT, RUN = audit.d, audit.ROOT, audit.RUN
RESULTS = ROOT/"results"/RUN
OUT = RESULTS/"materialization.json"


def reconstruct(index, desc, row, x, te, constants):
    audit.require(desc["id"] == row["id"] and desc["label"] == row["label"]
                  and f"{audit.FAULT}|bus:{int(desc['group'])}" == row["group"]
                  and desc["root"] == row["metadata"]["source_root"], "event identity differs")
    audit.require(x.dtype == te.dtype == np.float64 and x.shape == (row["points"],30)
                  and te.shape == (row["points"],10), "frozen array shape differs")
    t = np.asarray(desc["t"],dtype=np.float64) if "t" in desc else np.arange(len(x))*.01
    audit.require(abs(t[-1]-row["metadata"]["trajectory_end_s"]) < 1e-9, "event duration differs")
    tm = x[:,20:30]
    for key, values in (("theta",x[:,:10]),("omega",x[:,10:20]),("tau_m",tm),("tau_e",te)):
        audit.require(np.isfinite(values).sum(axis=0).tolist() == row["coverage"][key],
                      "frozen observation coverage differs")
    return d.event_record(row["id"],row["group"],row["label"],t,
                          x[:,:10]+np.asarray(constants["theta_ref"]),x[:,10:20],tm,tm,te,
                          row["metadata"],constants["mass"])


def gauge_check():
    x = np.arange(310,dtype=np.float64).reshape(31,10)/71
    mass = np.arange(1,11,dtype=np.float64)
    delta = np.arange(10,dtype=np.float64)/13
    theta_ref = delta-delta@(mass/mass.sum())
    center = lambda a: a-a[0]@(mass/mass.sum())
    np.testing.assert_allclose(center(x+delta),center(x+theta_ref),rtol=0,atol=2e-15)


def main():
    audit.require(not OUT.exists() and not audit.BUNDLE.exists(), "materialization already exists")
    start, cpu = time.perf_counter(), time.process_time()
    gauge_check()
    adoption_ref = audit.reference(RESULTS/"observation_adoption.json")
    adoption = d._bound_record(adoption_ref,RESULTS/"observation_adoption.json")
    audit.require(adoption["run_id"] == RUN and adoption["status"] == "adopted_for_prediction"
                  and adoption["materialization_released"] is True, "root materialization release required")
    for key, path in (("adapter",ROOT/"src/kiki/data/osgkd.py"),("materializer",Path(__file__).resolve())):
        audit.require(audit.reference(path)["sha256"] == adoption["source_hashes"][key], "adopted code differs")
    records = {key:d._bound_record(adoption[key],RESULTS/name) for key,name in (
        ("original_audit","port_audit.json"),("recheck","port_audit_recheck.json"),
        ("attribution","source_time_attribution.json"))}
    for key in ("design","review"):
        ref = adoption[key]
        d._bound_record_bytes(ref,ROOT/ref["path"])
    original, recheck, attribution = (records[k] for k in ("original_audit","recheck","attribution"))
    audit.require(original["status"] == recheck["status"] == "blocked"
                  and original["run"] == recheck["run"] == RUN
                  and attribution["run_id"] == RUN and attribution["status"] == "unresolved"
                  and recheck["original_audit"] == adoption["original_audit"]
                  and original["source_notes"]["cache_offset_difference"] == 0, "frozen evidence differs")
    for key in ("ordered_ids","fit_ids","check_ids","constants_id"):
        audit.require(original[key] == recheck[key], "original cohort differs")
    constants = original["constants"]
    audit.require(constants["constants_id"] == original["constants_id"], "constants identity differs")
    source = audit.reference(audit.CACHE,audit.HASHES[audit.CACHE])
    audit.require(source == original["sources"]["fit_events.npz"], "cache identity differs")
    events = []
    array_bytes = 0
    with np.load(audit.CACHE,allow_pickle=False) as cache:
        descriptions = json.loads(str(cache["metadata_json"]))
        audit.require(len(descriptions) == len(original["rows"]) == 297
                      and [e["id"] for e in descriptions] == original["ordered_ids"], "ordered cohort differs")
        expected = {"metadata_json"}|{f"event_{i:04d}_{key}" for i in range(297) for key in ("x","torque")}
        audit.require(set(cache.files) == expected, "cache members differ")
        for i,(desc,row) in enumerate(zip(descriptions,original["rows"],strict=True)):
            x, te = cache[f"event_{i:04d}_x"], cache[f"event_{i:04d}_torque"]
            array_bytes += x.nbytes+te.nbytes
            events.append(reconstruct(i,desc,row,x,te,constants))
    fit,check = d.split_events(events)
    audit.require([e["id"] for e in fit] == original["fit_ids"]
                  and [e["id"] for e in check] == original["check_ids"], "frozen split differs")
    bundle = audit.export(events,constants)
    receipt = dict(run_id=RUN,status="completed",observation_adoption=adoption_ref,
                   **{key:adoption[key] for key in ("original_audit","recheck","attribution")},
                   bundle=bundle,source=source,constants_id=constants["constants_id"],
                   **{key:original[key] for key in ("ordered_ids","fit_ids","check_ids")},
                   work=dict(cache_open_count=1,metadata_reads=1,event_array_reads=594,
                             event_array_bytes=array_bytes,source_hash_passes=1,ode_calls=0,
                             integrals=0,calibrations=0,val_test_reads=0,training_updates=0,
                             wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu))
    with OUT.open("xb") as stream:
        stream.write(d.canonical_json(receipt))
    print(json.dumps({"status":"completed","bundle":bundle,"receipt":audit.reference(OUT)}))


if __name__ == "__main__":
    main()
