"""Mechanical data QA only, including holdout; never fit or select a predictor."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.distance import pdist, squareform

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from kiki.data.private_dataset import canonical_arrays, check_arrays, los_index
from kiki.physics import coi_project


def audit(directory):
    directory = Path(directory)
    with (directory/"dataset_index.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    splits = json.loads((directory/"splits.json").read_text())
    ids = [row["event_id"] for row in rows]
    assert len(ids) == len(set(ids)) > 0
    assert set(ids) == {eid for values in splits.values() for eid in values}
    assert sum(map(len, splits.values())) == len(ids)
    byid = {row["event_id"]: row for row in rows}
    for grouping in ("fault_bus", "fault_line"):
        groups = {role: {byid[eid][grouping] for eid in values} for role, values in splits.items()}
        for a,b in (("train","validation"),("train","test"),("validation","test")):
            assert not groups[a] & groups[b], (grouping,a,b)
    coverage, pm_ranges, coi_errors, counts = [], [], [], Counter()
    with h5py.File(directory/"dataset.h5", "r") as file:
        assert set(file["events"]) == set(ids)
        cfg = json.loads(file.attrs["config"])
        mapping = json.loads(file.attrs["state_mapping"])
        assert [r[0] for r in mapping] == list(range(len(mapping)))
        inertia = file["inertia"][...]
        equilibrium = {key: file["equilibrium"][key][...] for key in ("delta","omega","Pm","Pe")}
        for row in rows:
            e = file["events"][row["event_id"]]
            arrays = {key: e[key][...] for key in ("t","delta","omega","Pm","Pe")}
            x = e["full_state"][...]
            check_arrays(arrays,x,cfg)
            assert x.shape[1] == len(mapping)
            assert e["algebraic_state"].shape == (len(x),len(json.loads(file.attrs["algebraic_names"])))
            for name in ("algebraic_state","bus_v","bus_a","terminal_v","Qe","te"):
                assert np.isfinite(e[name][...]).all(), name
            for key in ("delta","omega"):
                columns = [r[0] for r in mapping if r[1] == key and r[2] == "GENROU"]
                np.testing.assert_array_equal(x[:,columns], arrays[key])
            assert e.attrs["postfault_signature"] == file.attrs["postfault_signature"] == row["postfault_signature"]
            assert e.attrs["solver_success"] and e.attrs["failed_steps"] == 0
            assert int(row["valid_length"]) == len(x)
            trigger = los_index(arrays["delta"],arrays["omega"],inertia,cfg)
            if row["label"] == "unstable":
                assert row["termination_reason"] == "physical_loss_of_synchronism"
                assert trigger == len(x)-1
                assert abs(float(row["event_time_s"])-arrays["t"][trigger]) < 1e-9
            else:
                assert row["label"] == "stable" and row["termination_reason"] == "stable_horizon_reached"
                assert trigger is None and abs(arrays["t"][-1]-cfg["postfault_horizon_s"]) < 1e-9
                tail = round(cfg["stable_tail_seconds"]/cfg["sample_interval_s"])
                assert np.max(np.ptp(arrays["delta"][-tail:],axis=1)) < cfg["los_angle_rad"]
                assert np.max(abs(coi_project(arrays["omega"][-tail:],inertia))) < cfg["stable_tail_speed_pu"]
                pm_ranges.append(float(np.max(np.ptp(arrays["Pm"],axis=0))))
            data = canonical_arrays(arrays,equilibrium,inertia,file.attrs["omega_s"])
            residuals = [np.max(abs(data[k]@inertia))/sum(inertia) for k in ("delta","omega")]
            residuals += [np.max(abs(data[k].sum(axis=1))) for k in ("Pm","Pe")]
            coi_errors.append(max(residuals))
            assert max(residuals) < 1e-9
            coverage.append(np.concatenate([data[k][0] for k in ("delta","omega","Pm")]))
            counts[row["label"]] += 1
        attempted, rejected = int(file.attrs["attempted"]), json.loads(file.attrs["discard_counts"])
        assert attempted == len(rows) + sum(rejected.values())
    assert pm_ranges and max(pm_ranges) > 1e-4, "No observable stable Pm dynamics"
    states = np.asarray(coverage)
    scales = np.maximum(states.std(axis=0),1e-6)
    states = (states-states.mean(axis=0))/scales
    singular = np.linalg.svd(states,compute_uv=False)
    pca = (singular**2/max(float(sum(singular**2)),1e-30))[:3].tolist()
    distances = squareform(pdist(states))
    np.fill_diagonal(distances,np.inf)
    nearest = distances.min(axis=1)
    train_groups = len({byid[i]["fault_bus"] for i in splits["train"]})
    val_groups = len({byid[i]["fault_bus"] for i in splits["validation"]})
    pilot_ready = counts["stable"] > 0 and counts["unstable"] > 0 and train_groups >= 8 and val_groups >= 3
    text = (f"# Private dataset QA\n\nPASS: {len(rows)}/{attempted} accepted; discarded {sum(rejected.values())}.\n\n"
            f"Outcomes: {dict(counts)}. Dynamic states: {len(mapping)}.\n\n"
            f"Discard reasons: {rejected}.\n\n"
            f"Stable raw Pm peak-to-peak max: {max(pm_ranges):.8g} system pu.\n"
            f"Maximum COI residual: {max(coi_errors):.3g}.\n\n"
            f"Clearing-state PCA fractions (standardized, descriptive only): {pca}.\n"
            f"Nearest-neighbor standardized distance min/median: {nearest.min():.6g}/{np.median(nearest):.6g}.\n\n"
            f"Split counts: { {r:len(v) for r,v in splits.items()} }. Bus/line groups disjoint.\n"
            f"Pilot sufficient for M2/M3-Pm: {pilot_ready} (train groups {train_groups}, validation groups {val_groups}).\n\n"
            "Checks: restored-system signature, finite arrays, strict uniform time, no repeated/frozen states, "
            "core shapes, state mapping, uniform physical outcome, normal solver at LOS, COI, group isolation.\n"
            "Holdout accessed only by mechanical data QA/coverage, not by any model or selection routine.\n")
    (directory/"qa_report.md").write_text(text,encoding="utf-8")
    print(text)
    return pilot_ready


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("directory")
    audit(parser.parse_args().directory)
