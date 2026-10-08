"""Loop0077 fixed three-condition, four-arm numerical probe.

This runner is deliberately a small orchestration layer around the existing
TRAIN generator.  It reads only the three selected 0075 event payloads and
never writes a dataset root or changes split semantics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / ".loop-work" / "research_deps_0067"
if DEPS.is_dir():
    sys.path.insert(0, str(DEPS))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.check_train_boundary import compare_pair  # noqa: E402
from experiments.generate_private_dataset import simulate  # noqa: E402
import andes  # noqa: E402

EVENT_IDS = ("boundary_0000", "boundary_0002", "boundary_0015")
ARMS = (("repeat_0025_tol1e-7", 0.0025, 1e-7),
        ("refine_00125_tol1e-7", 0.00125, 1e-7),
        ("refine_000625_tol1e-7", 0.000625, 1e-7),
        ("tol_0025_tol1e-9", 0.0025, 1e-9))


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _load_source(root: Path) -> tuple[dict, dict[str, dict], dict[str, dict]]:
    splits = json.loads((root / "splits.json").read_text(encoding="utf-8"))
    if set(splits.get("validation", [])) or set(splits.get("test", [])):
        raise ValueError("0075 source contains non-TRAIN split entries")
    if not set(EVENT_IDS).issubset(set(splits.get("train", []))):
        raise ValueError("selected 0075 IDs are not all TRAIN")
    rows = {}
    with (root / "dataset_index.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            rows[row["event_id"]] = row
    old: dict[str, dict] = {}
    with h5py.File(root / "dataset.h5", "r") as handle:
        config = json.loads(_decode(handle.attrs["config"]))
        common = {"postfault_signature": _decode(handle.attrs.get("postfault_signature", "")),
                  "state_mapping": json.loads(_decode(handle.attrs.get("state_mapping", "[]"))),
                  "case_sha256": _decode(handle.attrs.get("case_sha256", "")),
                  "equilibrium_full_state": np.asarray(handle["equilibrium/full_state"][...]),
                  "inertia": np.asarray(handle["inertia"][...])}
        for event_id in EVENT_IDS:
            if event_id not in rows or event_id not in handle["events"]:
                raise ValueError(f"missing selected 0075 event {event_id}")
            group = handle["events"][event_id]
            attrs = {name: _decode(group.attrs[name]) for name in group.attrs}
            arrays = {name: np.asarray(group[name][...]) for name in group}
            old[event_id] = {"attrs": attrs, "arrays": arrays, "provenance": common}
    events = {}
    for event_id in EVENT_IDS:
        row = rows[event_id]
        events[event_id] = {"line": row["fault_line"], "location": float(row["fault_location"]),
                            "bus": int(row["fault_bus"]), "duration_s": float(row["fault_duration_s"]),
                            "xf": float(row["fault_xf"])}
    return config, events, old


def _metadata_attrs(metadata: dict) -> dict:
    return {str(k): v for k, v in metadata.items()}


def _repeat_gate(first: dict | None, second: dict | None, *, mass: np.ndarray, omega_s: float) -> dict:
    if first is None or second is None:
        return {"status": "UNKNOWN", "qualified": False, "missing_first": first is None,
                "missing_second": second is None}
    a, b = first["attrs"], second["attrs"]
    aa, bb = first["arrays"], second["arrays"]
    same_grid = "t" in aa and "t" in bb and np.array_equal(aa["t"], bb["t"])
    same_label = a.get("label") == b.get("label")
    same_origin = abs(float(a.get("record_start_delay_s", np.nan)) -
                     float(b.get("record_start_delay_s", np.nan))) <= 1e-8
    stop_a, stop_b = a.get("event_time_s", ""), b.get("event_time_s", "")
    same_stop = stop_a == stop_b
    metrics = compare_pair(first, second, mass=mass, omega_s=omega_s)
    metric_ok = all(metrics.get(name, np.inf) <= 1e-8 for name in
                    ("combined_rmse_normalized", "common_frequency_rmse_normalized"))
    result = {"status": "QUALIFIED" if same_grid and same_label and same_origin and same_stop and metric_ok else "UNKNOWN",
              "qualified": bool(same_grid and same_label and same_origin and same_stop and metric_ok),
              "checks": {"label": same_label, "grid": same_grid, "record_origin": same_origin,
                         "physical_stop_time": same_stop, "normalized_metrics": metric_ok},
              "metrics": metrics}
    return result


def _comparison(first, second, mass, omega_s):
    result = compare_pair(first, second, mass=mass, omega_s=omega_s)
    if first is not None and second is not None:
        a, b = first['arrays'], second['arrays']
        n = min(len(a['t']), len(b['t']))
        result['per_variable_raw_errors'] = {}
        for key in ('delta', 'omega', 'Pm', 'Pe'):
            error = a[key][:n] - b[key][:n]
            result['per_variable_raw_errors'][key] = {
                'initial_max_abs': float(np.max(np.abs(error[0]))),
                'shared_prefix_rmse': float(np.sqrt(np.mean(error**2)))}
    return result


def run(source_root: Path, output: Path, solver_workdir: Path | None = None) -> dict:
    if output.exists():
        raise ValueError(f"refusing to overwrite {output}")
    config, events, old = _load_source(source_root)
    starting_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    case_sha = hashlib.sha256(Path(andes.get_case(config['case'])).read_bytes()).hexdigest()
    if case_sha != old[EVENT_IDS[0]]['provenance']['case_sha256']:
        raise ValueError('ANDES case differs from source')
    source_hashes = {name: hashlib.sha256((source_root/name).read_bytes()).hexdigest()
                     for name in ('dataset.h5', 'dataset_index.csv', 'splits.json')}
    output.mkdir(parents=True)
    cases: dict[str, dict] = {}
    with h5py.File(source_root / "dataset.h5", "r") as handle:
        mass = np.asarray(handle["inertia"][...], dtype=float)
        omega_s = float(handle.attrs.get("omega_s", 2 * np.pi * 60.0))
    for event_id in EVENT_IDS:
        case = {"event_id": event_id, "arms": {}}
        for arm_name, step, tol in ARMS:
            arm_config = dict(config)
            arm_config["integration_step_s"] = step
            arm_config["newton_tolerance"] = tol
            diagnostics: dict[str, Any] = {}
            row: dict[str, Any] = {"event_id": event_id, "arm": arm_name,
                                   "integration_step_s": step, "newton_tolerance": tol,
                                   "event": events[event_id], "config": arm_config}
            started = time.perf_counter()
            print(event_id, arm_name, 'START', flush=True)
            try:
                arrays, metadata, equilibrium, inertia, mapping, _ = simulate(events[event_id], arm_config,
                                                 solver_workdir=solver_workdir,
                                                 diagnostics=diagnostics)
                np.savez_compressed(output / f"{event_id}__{arm_name}.npz", **arrays)
                provenance = {"postfault_signature": metadata.get("postfault_signature"),
                              "state_mapping": mapping, "equilibrium_full_state": equilibrium["full_state"].tolist(),
                              "inertia": np.asarray(inertia).tolist()}
                old_prov = old[event_id]["provenance"]
                provenance_checks = {
                    "postfault_signature": provenance["postfault_signature"] == old_prov["postfault_signature"],
                    "state_mapping": provenance["state_mapping"] == old_prov["state_mapping"],
                    "equilibrium_full_state": np.allclose(provenance["equilibrium_full_state"], old_prov["equilibrium_full_state"], atol=1e-9, rtol=0),
                    "inertia": np.array_equal(np.asarray(provenance["inertia"]), old_prov["inertia"]),
                }
                if not all(provenance_checks.values()):
                    raise ValueError(f"source_provenance_mismatch:{provenance_checks}")
                row.update({"status": "completed", "metadata": metadata,
                            "diagnostics": diagnostics, "provenance": provenance,
                            "provenance_checks": provenance_checks})
                case["arms"][arm_name] = {"attrs": _metadata_attrs(metadata), "arrays": arrays,
                                           "provenance": provenance}
                print(event_id, arm_name, metadata.get("label"), "completed", flush=True)
            except Exception as exc:  # preserve every preregistered attempt
                row.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                            "diagnostics": diagnostics})
                case["arms"][arm_name] = None
                print(event_id, arm_name, "FAILED", type(exc).__name__, flush=True)
            row['seconds'] = time.perf_counter()-started
            (output / f"{event_id}__{arm_name}.json").write_text(
                json.dumps(_jsonable(row), indent=2, allow_nan=False), encoding="utf-8")
        cases[event_id] = case

    comparisons = []
    for event_id in EVENT_IDS:
        arm_rows = cases[event_id]["arms"]
        repeat = arm_rows[ARMS[0][0]]
        tol_arm = arm_rows[ARMS[3][0]]
        comparisons.append({"event_id": event_id, "comparison": "old0075_fine_vs_new_repeat",
                            **_comparison(old[event_id], repeat, mass, omega_s)})
        comparisons.append({"event_id": event_id, "comparison": "old0075_fine_vs_new_repeat_reproducibility_gate",
                            **_repeat_gate(old[event_id], repeat, mass=mass, omega_s=omega_s)})
        comparisons.append({"event_id": event_id, "comparison": "repeat_vs_tight_tolerance",
                            **_comparison(repeat, tol_arm, mass, omega_s)})
        for left, right in ((ARMS[0][0], ARMS[1][0]), (ARMS[1][0], ARMS[2][0])):
            comparisons.append({"event_id": event_id, "comparison": f"{left}_vs_{right}",
                                **_comparison(arm_rows[left], arm_rows[right], mass, omega_s)})
    trends = []
    for event_id in EVENT_IDS:
        rows = [row for row in comparisons if row['event_id'] == event_id]
        previous, latest = rows[-2:]
        metrics = ('combined_rmse_normalized', 'common_frequency_rmse_normalized')
        ratios = {key: (latest[key]/previous[key] if previous.get(key, 0) > 0 and key in latest else None)
                  for key in metrics}
        nonincreasing = all(key in previous and key in latest and latest[key] <= previous[key]+1e-12
                            for key in metrics)
        trends.append({'event_id': event_id, 'last_over_previous_ratios': ratios,
                       'nonincreasing': nonincreasing,
                       'last_pair_qualified': latest.get('qualified', False)})
    source_hashes_after = {name: hashlib.sha256((source_root/name).read_bytes()).hexdigest()
                           for name in source_hashes}
    if source_hashes_after != source_hashes:
        raise RuntimeError('source modified during probe')
    report = {"loop": "0077", "source_root": str(source_root), "read_validation_or_test_payloads": False,
              "source_hashes": source_hashes, "source_hashes_after": source_hashes_after,
              "git_commit": starting_commit,
              "code_hashes": {name: hashlib.sha256((ROOT/'experiments'/name).read_bytes()).hexdigest()
                              for name in ('generate_private_dataset.py','probe_boundary_numerics.py','check_train_boundary.py')},
              "case_sha256": case_sha, "refinement_trends": trends,
              "source_config": config,
              "event_ids": list(EVENT_IDS), "arms": [{"name": n, "integration_step_s": s,
              "newton_tolerance": t} for n, s, t in ARMS], "attempts": 12,
              "cases": [{"event_id": eid, "arms": [{"arm": n, "status":
              cases[eid]["arms"][n] is not None} for n, _, _ in ARMS]} for eid in EVENT_IDS],
              "comparisons": comparisons}
    (output / "report.json").write_text(json.dumps(_jsonable(report), indent=2, allow_nan=False), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=ROOT / "private_data/ieee39_boundary_fine_0075")
    parser.add_argument("--output", type=Path, default=ROOT / "private_data/task2_boundary_probe_0077")
    parser.add_argument("--solver-workdir", type=Path, default=ROOT / ".loop-work/andes_boundary_probe0077")
    args = parser.parse_args()
    report = run(args.source_root, args.output, args.solver_workdir)
    print(json.dumps({"output": str(args.output), "attempts": report["attempts"],
                      "comparisons": len(report["comparisons"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
