"""Prospective complete train and original val packets; frozen evidence stays intact."""
from collections import Counter
import csv
import hashlib
import json

import h5py
import numpy as np

from . import osgkd as d

PREPARE_SHA256 = "a6f906a5e933f1a39af76546826851bd92e534b594e1aeaae3d027f85c64199a"
RELEASE_SHA256 = "02be50898c5702d5739931032ce2a79436d96294a7244911a26bdb1b01c225a5"
SHARDS = ("ieee39_postfault", "ieee39_expansion")
SOURCE_METADATA = {
    "ieee39_postfault": (
        "02990d0ddb22e4153824a28f1e99fc0d22a0afb70a4d197e6a294e6da253dba8",
        "3cb759749328986790fc6933c34aba416be10ef807135b4654f1ae970d84de9b",
    ),
    "ieee39_expansion": (
        "f1d12c0f60462c85991a41464f56fe247977cefaf5e458a22a4ecc0d7e61c10b",
        "6e4c0f3b5329f2472201869aa1034d2e2097726c400064a4cdf1bcc54ac3d783",
    ),
}
FAULT = "temporary_bus_three_phase_ground"
PM_REFERENCE_SHA256 = "8ee9052f7e1031ac2e5391ff2d59cf0da86acebae81a1b0e861563984fdead29"


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _metadata(requested):
    rows, splits = {}, {key: set() for key in ("train", "validation", "test")}
    for shard in SHARDS:
        root = d.ROOT / "private_data" / shard
        split_file, index_file = root / "splits.json", root / "dataset_index.csv"
        _require((_digest(split_file), _digest(index_file)) == SOURCE_METADATA[shard],
                 f"{shard} split or index identity differs")
        partition = json.loads(split_file.read_bytes())
        for key in splits:
            ids = partition[key]
            _require(len(ids) == len(set(ids)) and not splits[key].intersection(ids), "duplicate official IDs")
            splits[key].update(ids)
        shard_ids = set(partition[requested])
        with index_file.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                eid = row["event_id"]
                if eid in shard_ids:
                    _require(eid not in rows, "duplicate source metadata")
                    rows[eid] = {**row, "source_root": shard}
    _require(all(not splits[a] & splits[b] for a, b in
                 (("train", "validation"), ("train", "test"), ("validation", "test"))), "official split overlap")
    _require(set(rows) == splits[requested], "official metadata coverage differs")
    return rows, splits


def _optional(value):
    return None if value in (None, "", "None") else float(value)


def _event_meta(row, end):
    return dict(true_event_time_record_s=_optional(row.get("event_time_s")),
                record_start_delay_s=_optional(row.get("record_start_delay_s")),
                clearing_time_s=_optional(row.get("clearing_time_s")),
                termination_reason=row.get("termination_reason") or None,
                trajectory_end_s=float(end), source_root=row["source_root"])


def load_train():
    """Return all 385 authorized train events and unchanged train-only parent constants."""
    release = d.ROOT / "results" / d.RUN / "port_audit_release.json"
    _require(_digest(release) == RELEASE_SHA256, "frozen release identity differs")
    retained, constants = d.load_fit297(dict(path=release.relative_to(d.ROOT).as_posix(),
                                           sha256=RELEASE_SHA256, bytes=release.stat().st_size))
    constants = {k: v for k, v in constants.items() if k not in ("fit_ids", "check_ids")}
    root = d.ROOT / "private_data/r_memory_20261025/run01_repair01"
    _require(_digest(root / "prepare.json") == PREPARE_SHA256, "train prepare identity differs")
    prepare = json.loads((root / "prepare.json").read_bytes())
    cohort = {row["id"]: row for row in prepare["cohort"]}
    rows, splits = _metadata("train")
    retained = {event["id"]: event for event in retained}
    events = []
    for filename, ids_key, hash_key in (("fit_events.npz", "fit_ids", "fit_cache_sha256"),
                                       ("no_update_events.npz", "no_update_ids", "no_update_cache_sha256")):
        path = root / filename
        _require(_digest(path) == prepare[hash_key], "train cache identity differs")
        with np.load(path, allow_pickle=False) as cache:
            descriptions = json.loads(str(cache["metadata_json"]))
            _require([e["id"] for e in descriptions] == prepare[ids_key], "train cache IDs differ")
            for index, desc in enumerate(descriptions):
                eid = desc["id"]
                x, te = (cache[f"event_{index:04d}_{key}"] for key in ("x", "torque"))
                _require(eid in cohort and desc["label"] == cohort[eid]["label"]
                         and desc["group"] == cohort[eid]["group"], "train cohort identity differs")
                _require(x.dtype == te.dtype == np.float64 and x.shape == (len(x), 30)
                         and te.shape == (len(x), 10) and len(x) >= 12, "train array shape differs")
                cm = desc["classification_meta"]
                _require(np.allclose(cm["inertia"], np.asarray(constants["mass"])*constants["omega_s"])
                         and cm["omega_s"] == constants["omega_s"], "train physical reference differs")
                reference = np.asarray(cm["equilibrium_delta"])
                mass = np.asarray(constants["mass"])
                _require(np.allclose(reference-reference@(mass/mass.sum()), constants["theta_ref"], atol=1e-12),
                         "train angular reference differs")
                t = np.asarray(desc.get("t", np.arange(len(x))*d.DT), dtype=np.float64)
                if eid in rows:
                    row = rows[eid]
                    _require(eid in splits["train"] and desc["root"] == row["source_root"]
                             and row["label"] == desc["label"] and int(row["valid_length"]) == len(x)
                             and int(row["fault_bus"]) == int(desc["group"])
                             and abs(float(row["trajectory_end_s"])-t[-1]) < 1e-8,
                             "train source identity differs")
                else:
                    _require(desc["root"] == "closure_discovery" and desc["new_split"] == "train"
                             and cm["system_attestation_complete"], "unapproved train source")
                    row = {**desc, "source_root": desc["root"], "event_time_s": cm.get("event_time_s")}
                meta = _event_meta(row, t[-1])
                event = d.event_record(eid, f"{FAULT}|bus:{int(desc['group'])}", desc["label"], t,
                                       x[:, :10]+constants["theta_ref"], x[:, 10:20], x[:, 20:],
                                       x[:, 20:], te, meta, constants["mass"])
                if eid in retained:
                    old = retained[eid]
                    for key in ("theta", "omega"):
                        np.testing.assert_array_equal(event["history"][key], old["history"][key])
                    for key in d.PORTS:
                        np.testing.assert_array_equal(event["targets"][key], old["targets"][key])
                events.append(event)
    events.sort(key=lambda event: event["id"])
    _require(len(events) == len(cohort) == len({e["id"] for e in events}) == 385
             and {e["id"] for e in events} == set(cohort)
             and Counter(e["label"] for e in events) == {"stable": 342, "unstable": 43}, "train denominator differs")
    return events, constants


def load_val(constants):
    """Return the entire original val split, retaining unavailable histories as None."""
    rows, splits = _metadata("validation")
    events = []
    common_reference = None
    for shard in SHARDS:
        with h5py.File(d.ROOT / "private_data" / shard / "dataset.h5", "r") as file:
            config = json.loads(file.attrs["config"])
            for source, key in (("omega_s", "omega_s"), ("mva_base", "mva_base"),
                                ("case_sha256", "source_case_sha256"), ("andes_version", "source_andes_version"),
                                ("raw_units", "raw_units"), ("postfault_signature", "postfault_signature")):
                _require(file.attrs[source] == constants[key], f"val {source} differs")
            _require(config["sample_interval_s"] == d.DT and config["integration_step_s"] == .005
                     and config["fault_type"] == FAULT
                     and config["disable_builtin_timed_events"] == constants["common_controls"]["disable_builtin_timed_events"]
                     and config["disable_pss"] == constants["common_controls"]["disable_pss"], "val controls differ")
            mass = np.asarray(constants["mass"])
            reference = {key: file["equilibrium"][key][...] for key in ("delta", "omega", "Pm")}
            _require(all(value.shape == (10,) and np.isfinite(value).all() for value in reference.values()),
                     "invalid val equilibrium")
            _require(hashlib.sha256(np.asarray(reference["Pm"], dtype="<f8").tobytes(order="C")).hexdigest()
                     == PM_REFERENCE_SHA256, "val/train common torque reference differs")
            if common_reference is not None:
                _require(all(np.array_equal(value, common_reference[key]) for key, value in reference.items()),
                         "val shards have different fixed references")
            common_reference = reference
            delta = reference["delta"]
            mapping = json.loads(file.attrs["state_mapping"])
            generator_states = [[state, "GENROU", generator]
                                for state in ("delta", "omega") for generator in constants["generator_order"]]
            _require([entry[1:] for entry in mapping[:20]] == generator_states, "val generator order differs")
            _require(np.allclose(file["inertia"][...], mass*constants["omega_s"], rtol=1e-12, atol=1e-10)
                     and np.allclose(delta-delta@(mass/mass.sum()), constants["theta_ref"], rtol=0, atol=1e-12)
                     and np.allclose(reference["omega"], 1., rtol=0, atol=1e-12), "val common reference differs")
            for eid in sorted(splits["validation"]):
                row = rows[eid]
                if row["source_root"] != shard:
                    continue
                raw = {key: file["events"][eid][key][...] for key in ("t", "delta", "omega", "Pm", "te")}
                t = raw["t"]
                _require(t.ndim == 1 and len(t) == int(row["valid_length"]) and len(t) > 0
                         and np.isfinite(t).all() and abs(t[0]) < 1e-9
                         and np.allclose(np.diff(t), d.DT, atol=1e-9, rtol=0)
                         and abs(t[-1]-float(row["trajectory_end_s"])) < 1e-8, "val timing differs")
                _require(all(raw[k].shape == (len(t), 10) and np.isfinite(raw[k]).all()
                             for k in ("delta", "omega", "Pm", "te")), "invalid val arrays")
                _require(row["label"] in ("stable", "unstable") and row["fault_type"] == FAULT
                         and row["postfault_signature"] == constants["postfault_signature"], "val source labels differ")
                meta = _event_meta(row, t[-1])
                _require(row["label"] != "unstable" or meta["true_event_time_record_s"] is not None
                         and abs(meta["true_event_time_record_s"]-t[-1]) < 1e-8, "val event time differs")
                group = f"{FAULT}|bus:{int(row['fault_bus'])}"
                if len(t) < 11:
                    event = dict(id=eid, group=group, label=row["label"], history=None, metadata=meta,
                                 missing_reason="fewer_than_11_original_history_points", original_points=len(t))
                else:
                    event = d.event_record(eid, group, row["label"], t, raw["delta"],
                                           (raw["omega"]-1)*constants["omega_s"], raw["Pm"]-reference["Pm"],
                                           raw["Pm"]-reference["Pm"], raw["te"]-reference["Pm"], meta, mass)
                events.append(event)
    events.sort(key=lambda event: event["id"])
    _require(len(events) == 117 and Counter(e["label"] for e in events) == {"stable": 95, "unstable": 22}
             and sum(e["history"] is None for e in events) == 10, "original val denominator differs")
    return events
