"""One bounded fit297 audit using the existing ordered torque/motion cache."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import time
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import h5py
import numpy as np

from kiki.data import osgkd as d

ROOT, RUN = d.ROOT, d.RUN
OUT = ROOT/"results"/RUN/"port_audit.json"
BUNDLE = ROOT/"artifacts"/RUN/"port_audit"/"data.npz"
FREEZE = ROOT/"results/gftd_matched_prediction_20261109/deploy/initialization_freeze.json"
CACHE = ROOT/"private_data/r_memory_20261025/run01_repair01/fit_events.npz"
CASE = ROOT/"private_data/gftd_fit297_20261107/reference/ieee39_full.xlsx"
BASE = ROOT/"private_data/task2_m3_0020/model.npz"
PROTOCOL = ROOT/"campaigns/osgkd_ieee39/first_fit297_protocol.md"
SHARDS = ("ieee39_postfault","ieee39_expansion")
HASHES = {FREEZE:"5743dba4aa7fd1e00da79745ac01c9d22e1b50f1ed0259fb49711596eb458e04",
          CACHE:"3058c06fe23479a2c494203bd379ca495c54953b6d8b993a6591c26905cc4252",
          CASE:"9c2048dc94201ee48ffe816de65d53831fa1f4b0e7fdf32f24c50f1017edcfe5"}
ORDER = "9a76d08d0423302a2b07374e9f2a92646db3be089cadeebaadace5b985699de4"
FAULT = "temporary_bus_three_phase_ground"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def reference(path, expected=None):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream,"sha256").hexdigest()
    require(expected is None or digest == expected, f"file identity differs: {path.name}")
    return dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size, sha256=digest)


def whitelist():
    source = json.loads(FREEZE.read_bytes())["rows"]
    def rows(mode):
        return [(e["index"],e["event_id"],e["label"]) for e in source[mode]]
    entries = rows("static")
    ids = [row[1] for row in entries]
    require(entries == rows("modulated") and len(ids) == len(set(ids)) == 297
            and [r[0] for r in entries] == list(range(297))
            and Counter(r[2] for r in entries) == {"stable":267,"unstable":30}
            and d.sha_bytes(d.canonical_json(ids)) == ORDER,"fit297 ordered whitelist differs")
    return entries


def source_machines():
    ns = {"s":"http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(CASE) as file:
        sheets = ET.fromstring(file.read("xl/workbook.xml")).findall("s:sheets/s:sheet",ns)
        sheet, = [s for s in sheets if s.get("name")=="GENROU"]
        rels = {r.get("Id"):r.get("Target") for r in ET.fromstring(file.read("xl/_rels/workbook.xml.rels"))}
        target = rels[sheet.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")]
        shared = (["".join(s.itertext()) for s in ET.fromstring(file.read("xl/sharedStrings.xml")).findall("s:si",ns)]
                  if "xl/sharedStrings.xml" in file.namelist() else [])
        root = ET.fromstring(file.read(target.lstrip("/") if target.startswith("/") else "xl/"+target))
        rows = []
        for row in root.findall("s:sheetData/s:row",ns):
            result = {}
            for cell in row.findall("s:c",ns):
                val,kind = cell.find("s:v",ns),cell.get("t")
                col = "".join(ch for ch in cell.get("r") if ch.isalpha())
                if kind == "inlineStr":
                    result[col] = "".join(cell.find("s:is",ns).itertext())
                elif val is not None:
                    result[col] = shared[int(val.text)] if kind=="s" else val.text
            rows.append(result)
    machines = [{name:row[k] for k,name in rows[0].items() if name in ("idx","bus","u","Sn","M","D","fn")}
                for row in rows[1:] if row.get("A")]
    require([m["idx"] for m in machines] == [f"GENROU_{n}" for n in range(1,11)]
            and [int(m["bus"]) for m in machines] == list(range(30,40))
            and all(float(m["u"]) == 1 and float(m["fn"]) == 60 for m in machines),"source machine identity differs")
    return machines


def common_reference():
    refs = []
    for shard in SHARDS:
        with h5py.File(ROOT/"private_data"/shard/"dataset.h5","r") as file:
            config = json.loads(file.attrs["config"])
            ref = {k:file.attrs[k] for k in ("omega_s","mva_base","case_sha256","andes_version","raw_units","postfault_signature")}
            ref.update({k:config[k] for k in ("fault_type","disable_builtin_timed_events","disable_pss","sample_interval_s","integration_step_s")})
            ref.update({k:file["equilibrium"][k][...] for k in ("delta","omega","Pm")})
            ref["inertia"] = file["inertia"][...]
            refs.append(ref)
    ref,other = refs
    require(all(np.array_equal(v,other[k]) for k,v in ref.items()),"common physical reference differs")
    omega_s = float(ref["omega_s"])
    require(omega_s == 2*np.pi*60 and ref["mva_base"]==100 and ref["case_sha256"]==HASHES[CASE]
            and ref["andes_version"]=="2.0.0" and ref["fault_type"]==FAULT
            and ref["disable_builtin_timed_events"] and ref["sample_interval_s"]==.01
            and ref["integration_step_s"]==.005 and np.allclose(ref["omega"],1.,rtol=0,atol=1e-12)
            and "Pm=GENROU.tm" in ref["raw_units"] and "te=GENROU.te" in ref["raw_units"],"unexpected source contract")
    machines = source_machines()
    base = np.array([float(m["Sn"])/100 for m in machines])
    require(np.allclose(ref["inertia"],np.array([float(m["M"]) for m in machines])*base,atol=1e-10,rtol=1e-12),
            "source inertia system-base conversion differs")
    mass = ref["inertia"]/omega_s
    damping = np.array([float(m["D"]) for m in machines])*base/omega_s
    with np.load(BASE,allow_pickle=False) as saved:
        old_offset = saved["pm_star"]
    require(old_offset.shape==(10,) and np.isfinite(old_offset).all(),"invalid legacy tm offset")
    theta_ref = ref["delta"] - ref["delta"]@(mass/mass.sum())
    constants = dict(mass=mass.tolist(),damping=damping.tolist(),F=np.eye(10).tolist(),theta_ref=theta_ref.tolist(),
                     omega_s=omega_s,mva_base=float(ref["mva_base"]),generator_order=[m["idx"] for m in machines],
                     generator_buses=list(range(30,40)),candidate_graph="undirected_complete_K10",
                     postfault_signature=str(ref["postfault_signature"]),source_case_sha256=str(ref["case_sha256"]),
                     source_andes_version=str(ref["andes_version"]),raw_units=str(ref["raw_units"]),
                     common_controls={"disable_builtin_timed_events":True,"disable_pss":bool(ref["disable_pss"]),
                                      "condition":"fixed restored-system configuration and original controller setpoints"})
    return ref,constants,old_offset


def selected_metadata(ids):
    rows,refs = {},{}
    for shard in SHARDS:
        path = ROOT/"private_data"/shard/"dataset_index.csv"
        refs[shard] = reference(path)
        with path.open(encoding="utf-8",newline="") as stream:
            for row in csv.DictReader(stream):
                if row["event_id"] in ids:
                    require(row["event_id"] not in rows,"duplicate selected source metadata")
                    rows[row["event_id"]] = row
    require(set(rows)=={eid for eid in ids if not eid.startswith("newtrain_")},"historical source coverage differs")
    return rows,refs


def optional(value):
    return None if value in (None,"","None") else float(value)


def inspect_event(cache,i,desc,selected,ref,constants,offset):
    eid,label = desc["id"],desc["label"]
    classification = desc["classification_meta"]
    require(np.array_equal(ref["delta"],classification["equilibrium_delta"])
            and np.array_equal(ref["inertia"],classification["inertia"])
            and ref["omega_s"] == classification["omega_s"],"event common reference differs")
    old = selected.get(eid)
    if old is not None:
        require(old["label"]==label and old["fault_type"]==FAULT and int(old["fault_bus"])==int(desc["group"])
                and old["postfault_signature"]==ref["postfault_signature"],"historical source identity differs")
    else:
        require(eid.startswith("newtrain_") and desc["root"]=="closure_discovery" and desc["new_split"]=="train"
                and classification["system_attestation_complete"],"new source fixed-system attestation missing")
    prefix = f"event_{i:04d}"
    x,te = cache[prefix+"_x"],cache[prefix+"_torque"]
    require(x.dtype==te.dtype==np.float64 and x.ndim==2 and x.shape[1]==30 and te.shape==(len(x),10),"cache shape differs")
    t = np.asarray(desc["t"],dtype=np.float64) if "t" in desc else np.arange(len(x),dtype=np.float64)*.01
    if old is not None:
        require(int(old["valid_length"])==len(x) and abs(float(old["trajectory_end_s"])-t[-1])<=1e-8,"old record length differs")
    theta,omega = x[:,:10]+ref["delta"],x[:,10:20]
    tm,te = x[:,20:]+(offset-ref["Pm"]),te+(offset-ref["Pm"])
    aux = old if old is not None else desc
    event_time = optional(aux.get("event_time_s",classification.get("event_time_s")))
    require(label!="unstable" or event_time is not None and abs(event_time-t[-1])<=1e-8,"true event timing differs")
    meta = dict(true_event_time_record_s=event_time,record_start_delay_s=optional(aux.get("record_start_delay_s")),
                clearing_time_s=optional(aux.get("clearing_time_s")),termination_reason=aux.get("termination_reason") or None,
                trajectory_end_s=float(t[-1]),source_root=desc["root"])
    group = f"{FAULT}|bus:{int(desc['group'])}"
    mass,damping = np.asarray(constants["mass"]),np.asarray(constants["damping"])
    event = d.event_record(eid,group,label,t,theta,omega,tm,tm,te,meta,mass)
    history = d.balance_window(t,omega,tm,te,mass,damping,0.,.1)
    future = d.balance_window(t,omega,tm,te,mass,damping,.1,min(1.1,float(t[-1])))
    row = dict(id=eid,label=label,group=group,points=len(x),future_s=float(event["t"][-1]),input_eligible=True,
               windows={"history":history,"early_future":future},metadata=meta,
               coverage={key:np.isfinite(value).sum(axis=0).tolist()
                         for key,value in (("theta",theta),("omega",omega),("tau_m",tm),("tau_e",te))})
    return event,row


def export(events,constants):
    arrays,descriptors = {},[]
    for i,event in enumerate(events):
        prefix = f"event_{i:04d}_"
        descriptors.append({k:event[k] for k in ("id","group","label","metadata")})
        arrays[prefix+"t"] = event["t"]
        arrays.update({prefix+"history_"+k:v for k,v in event["history"].items()})
        arrays.update({prefix+k:v for k,v in event["targets"].items()})
        arrays.update({prefix+"mask_"+k:v for k,v in event["masks"].items()})
    arrays["constants_json"] = np.asarray(d.canonical_json(constants).decode())
    arrays["events_json"] = np.asarray(d.canonical_json(descriptors).decode())
    BUNDLE.parent.mkdir(parents=True,exist_ok=True)
    with BUNDLE.open("xb") as stream:
        np.savez_compressed(stream,**arrays)
    return reference(BUNDLE)


def main():
    require(not OUT.exists() and not BUNDLE.exists(),"audit already has a receipt or bundle")
    start = time.perf_counter()
    receipt = dict(run=RUN,status="interrupted",rows=[],models_called=0,ode_calls=0,val_test_reads=0)
    try:
        receipt["sources"] = {path.name:reference(path,digest) for path,digest in HASHES.items()}
        receipt["sources"].update(common_legacy_tm_offset=reference(BASE),protocol=reference(PROTOCOL),
                                  adapter=reference(ROOT/"src/kiki/data/osgkd.py"),audit=reference(Path(__file__).resolve()))
        white = whitelist()
        ids = [row[1] for row in white]
        receipt.update(ordered_ids=ids,event_order_sha256=ORDER,denominators={"all":297,"stable":267,"unstable":30})
        selected,receipt["source_index_refs"] = selected_metadata(set(ids))
        ref,constants,offset = common_reference()
        events = []
        with np.load(CACHE,allow_pickle=False) as cache:
            descriptions = json.loads(str(cache["metadata_json"]))
            require([(i,e["id"],e["label"]) for i,e in enumerate(descriptions)]==white,"ordered cache differs")
            expected = {"metadata_json"}|{f"event_{i:04d}_{key}" for i in range(297) for key in ("x","torque")}
            require(set(cache.files)==expected,"unexpected cache members")
            for i,e in enumerate(descriptions):
                event,row = inspect_event(cache,i,e,selected,ref,constants,offset)
                events.append(event)
                receipt["rows"].append(row)
        fit,check = d.split_events(events)
        scales,counts = d.calibrate(fit)
        constants.update({k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in scales.items()})
        constants["constants_id"] = d.sha_bytes(d.canonical_json(constants))
        receipt.update(constants=constants,constants_id=constants["constants_id"],calibration_counts=counts,
                       group_order=d.group_order(events),fit_ids=[e["id"] for e in fit],check_ids=[e["id"] for e in check],
                       split_counts={name:dict(Counter(e["label"] for e in part)) for name,part in (("fit",fit),("check",check))},
                       source_notes=dict(mechanical="GENROU tm numeric torque on 100 MVA system base",
                           electrical="GENROU te air-gap torque on same base; Pe not present and not a substitute",
                           mapping="M=inertia/Omega, D=source-system-base-D/Omega, F=I in the source tm coordinate",
                           cache_offset_difference=float(np.max(np.abs(offset-ref["Pm"]))),
                           source_equation="https://raw.githubusercontent.com/CURENT/andes/v2.0.0/andes/models/synchronous/genbase.py",
                           electrical_provenance="reports/history/private_rom/originals/accepted/0048_electrical_port_balance.md",
                           optional_environment="No load, voltage or environment component labels constructed."))
        receipt["window_statuses"] = {key:dict(Counter(r["windows"][key]["status"] for r in receipt["rows"]))
                                      for key in ("history","early_future")}
        receipt["maximum_actual_unstable_future_s"] = max(r["future_s"] for r in receipt["rows"] if r["label"]=="unstable")
        blocked = [r["id"] for r in receipt["rows"] if r["windows"]["history"]["status"]!="passed"
                   or r["windows"]["early_future"]["status"] not in ("passed","not_applicable")]
        receipt.update(status="blocked" if blocked else "passed",balance_blocked_ids=blocked)
        if not blocked:
            receipt["bundle"] = export(events,constants)
    except Exception as error:
        receipt.update(error_type=type(error).__name__,error=str(error))
        raise
    finally:
        receipt["wall_seconds"] = time.perf_counter()-start
        OUT.parent.mkdir(parents=True,exist_ok=True)
        with OUT.open("xb") as stream:
            stream.write(d.canonical_json(receipt))
    print(json.dumps({k:receipt[k] for k in ("status","window_statuses","split_counts","balance_blocked_ids",
                                            "maximum_actual_unstable_future_s")},ensure_ascii=False))


if __name__ == "__main__":
    main()
