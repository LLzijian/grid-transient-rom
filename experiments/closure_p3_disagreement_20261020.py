"""One bounded, train-only P3 fixed-endpoint disagreement inference."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
KEYS = tuple(f"{seed}_{structure}" for seed in (6101, 6102, 6103)
             for structure in ("full8", "delete7"))
P2 = ROOT / "private_data/closure_mechanism_20261019/P2/fixed_edge_pair01/campaign01"
FINAL = ROOT / "results/fixed_edge_pair_20261019/final.json"
S_DELTA = .47675314588683515
S_OMEGA = 1.0033667249602924
EPSILON = 1e-6
LIMIT = .001
FUTURE_RUNNER_SHA = "6f3194387f13a103a04bd73d27dbe17cde7e786d8617655ba2fff8bf811d72c3"
P2_RUNNER_SHA = "4412777530c9a40f80641faf431991e2c62f38194c15023c8d6f1417ce4c14f9"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def local_sources():
    """Hash the transitive local Python import graph without importing it."""
    pending = [Path(__file__), ROOT / "tests/test_closure_p3_disagreement_20261020.py",
               ROOT / "experiments/fixed_edge_pair_20261019.py"]
    found = set()
    while pending:
        path = pending.pop().resolve()
        if path in found:
            continue
        if not path.is_file() or not path.is_relative_to(ROOT):
            raise ValueError(f"missing source: {path}")
        found.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if node.level:
                    base = ROOT / ("src" if path.is_relative_to(ROOT / "src") else "experiments")
                    parts = list(path.relative_to(base).parent.parts)
                    mod = ".".join(parts[:len(parts)-node.level+1] + ([mod] if mod else []))
                names = [mod] + [mod + "." + a.name for a in node.names if a.name != "*"]
            else:
                continue
            for name in names:
                parts = name.split(".")
                for base in (ROOT / "experiments", ROOT / "src"):
                    candidates = [base.joinpath(*parts).with_suffix(".py")]
                    candidates += [base.joinpath(*parts[:i]) / "__init__.py" for i in range(1, len(parts)+1)]
                    pending.extend(p for p in candidates if p.is_file())
    extras = [FINAL, ROOT / "reports/astra6_p3_admissibility_20261020.md",
              ROOT / "reports/p3_disagreement_20261020_protocol.md",
              ROOT / "private_data/closure_mechanism_20261019/P3/disagreement01/native_supervise.ps1"]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(found | set(extras))}


def frozen_inputs():
    final = json.loads(FINAL.read_text(encoding="utf-8"))
    if final.get("source_commit") != "47ba6f341886aeb4cc235d3c25a9b16bc8f4a24b":
        raise ValueError("historical P2 source commit changed")
    source = local_sources()
    expected = final["checkpoint_checks"]
    models = {}
    endpoint_evidence = {}
    for key in KEYS:
        folder = P2 / key / "checkpoints/step001024"
        meta = json.loads((folder / "checkpoint.json").read_text(encoding="utf-8"))
        checks = {name: sha(folder / name) for name in ("checkpoint.json", "model.json", "training.pt")}
        if (meta.get("step") != 1024 or checks["checkpoint.json"] != expected[key]["latest"]["checkpoint_sha256"]
                or checks["model.json"] != meta["model_sha256"]
                or checks["training.pt"] != meta["training_sha256"]):
            raise ValueError(f"frozen checkpoint identity mismatch: {key}")
        models[key] = checks
        complete_paths = list((P2 / key / "milestone_1024").glob("attempt_*/complete.json"))
        if len(complete_paths) != 1:
            raise ValueError(f"ambiguous final P2 evidence: {key}")
        complete = json.loads(complete_paths[0].read_text(encoding="utf-8"))
        provenance = complete["provenance"]
        if (complete["status"] != "COMPLETED" or provenance["checkpoint"] != str(folder / "model.json")
                or provenance["model_sha256"] != checks["model.json"]
                or provenance["seed"] != int(key[:4]) or provenance["structure"] != key[5:]
                or provenance["step"] != 1024 or provenance["split"] != "train"
                or provenance["identity"] != meta["identity"]
                or sha(complete_paths[0].parent / "evidence.json") != complete["evidence_sha256"]):
            raise ValueError(f"P2 final evidence provenance mismatch: {key}")
        endpoint_evidence[key] = {"complete": complete_paths[0].relative_to(ROOT).as_posix(),
                                  "complete_sha256": sha(complete_paths[0]),
                                  "evidence_sha256": complete["evidence_sha256"]}
        old_code = meta["identity"]["common"]["code_sha256"]
        for name, old_hash in old_code.items():
            current = sha(ROOT / name)
            if name == "experiments/fixed_edge_pair_20261019.py":
                if old_hash != P2_RUNNER_SHA or current != FUTURE_RUNNER_SHA:
                    raise ValueError("future runner is not the registered PR11-only difference")
            elif current != old_hash:
                raise ValueError(f"historical P2 dependency changed outside future runner: {name}")
            source[name] = current
        for name, old_hash in meta["identity"]["common"]["bound_files"].items():
            if Path(name).name == "dataset.h5":
                raise ValueError("historical whole-HDF5 input is forbidden in P3 freeze")
            if sha(name) != old_hash:
                raise ValueError(f"historical bound input changed: {name}")
    data = []
    # Historical dataset.h5 contains other splits: do not hash or open it here.
    # The original bound files and explicit train cohort are checked separately.
    for root_name in ("ieee39_postfault", "ieee39_expansion"):
        data += [ROOT / "private_data" / root_name / name
                 for name in ("dataset_index.csv", "splits.json")]
    train = ROOT / "private_data/closure_discovery_20261017/D/new_data/train"
    data += [p for p in train.rglob("*") if p.is_file()]
    from_source = ROOT / "private_data/task2_prediction_optimization_0102/formal_S_O/step00650/structured_numeric_export.json"
    data += [from_source]
    historical_bound = json.loads((P2 / "6101_full8/checkpoints/step001024/checkpoint.json").read_text(encoding="utf-8"))["identity"]["common"]["bound_files"]
    data += [Path(name) for name in historical_bound]
    data = sorted(set(data))
    return {"schema": 1, "historical_science_commit": final["source_commit"],
            "historical_runner_sha256": P2_RUNNER_SHA,
            "prospective_runner_sha256": FUTURE_RUNNER_SHA, "source": source,
            "models": models, "endpoint_evidence": endpoint_evidence,
            "data": {p.resolve().relative_to(ROOT).as_posix(): sha(p) for p in data}}


def verify_manifest(manifest):
    if frozen_inputs() != manifest:
        raise ValueError("frozen source/model/data identity changed")


def observable(path, mass):
    import numpy as np
    a = np.asarray(path, dtype=np.float64)
    m = np.asarray(mass, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 40 or m.shape != (10,) or not np.isfinite(a).all():
        raise ValueError("nonfinite or malformed 40-state path/mass")
    if not np.isfinite(m).all() or (m <= 0).any():
        raise ValueError("invalid inertia")
    theta = a[:, :10] - (a[:, :10] * m).sum(axis=1, keepdims=True) / m.sum()
    omega_bar = (a[:, 10:20] * m).sum(axis=1, keepdims=True) / m.sum()
    v = a[:, 10:20] - omega_bar
    return np.concatenate((theta / (S_DELTA * math.sqrt(40)),
                           v / (S_OMEGA * math.sqrt(40)),
                           omega_bar / (S_OMEGA * math.sqrt(2))), axis=1)


def disagreement(paths, mass):
    """paths maps six keys to equal-length physical 40-state paths, j=1..N."""
    import numpy as np
    arrays = {k: observable(paths[k], mass) for k in KEYS}
    shapes = {a.shape for a in arrays.values()}
    if len(shapes) != 1 or next(iter(shapes))[0] < 1:
        raise ValueError("missing or mismatched prediction times")
    result = {}
    for name, sl in (("theta", slice(0, 10)), ("relative_omega", slice(10, 20)),
                     ("common_omega", slice(20, 21)), ("total", slice(None))):
        f = np.stack([arrays[f"{s}_full8"][:, sl] for s in (6101, 6102, 6103)])
        d = np.stack([arrays[f"{s}_delete7"][:, sl] for s in (6101, 6102, 6103)])
        mf, md = f.mean(0), d.mean(0)
        b = float(np.mean(np.sum((mf - md) ** 2, axis=1) / 4))
        wf = np.sum((f - mf) ** 2, axis=(0, 2)) / 2
        wd = np.sum((d - md) ** 2, axis=(0, 2)) / 2
        w = float(np.mean((wf + wd) / 2))
        result[name] = {"B": b, "W": w}
    result["total"]["D"] = result["total"]["B"] / (result["total"]["W"] + EPSILON)
    return result


def numerical(fine, coarse, history, event, meta, mass, n):
    import numpy as np
    from evaluate_structured_rom import _pair_row, _safe
    if fine.shape != (1001, 40) or coarse.shape != (501, 40):
        raise ValueError("rollout output length or state dimension changed")
    if not 1 <= n <= 500 or not np.isfinite(fine[:2*n+1]).all() or not np.isfinite(coarse[:n+1]).all():
        raise ValueError("nonfinite actual-window fine 40-state substep or coarse state")
    # _pair_row prepends history[:10] itself; slicing preserves the actual t=10 origin.
    row = _pair_row(fine[:2*n+1], coarse[:n+1], history, event, meta, mass)
    a, b = fine[:2*n+1:2], coarse[:n+1]
    de, we = a[:, :10] - b[:, :10], a[:, 10:20] - b[:, 10:20]
    coi = de - (de * mass).sum(axis=1, keepdims=True) / mass.sum()
    wcoi = we - (we * mass).sum(axis=1, keepdims=True) / mass.sum()
    common = (we * mass).sum(axis=1) / mass.sum()
    coi_norm = float(np.sqrt(.5 * np.mean((coi/S_DELTA)**2 + (wcoi/S_OMEGA)**2)))
    common_norm = float(np.sqrt(np.mean((common/S_OMEGA)**2)))
    row["actual_window_N"] = n
    row["actual_window_coi_normalized_rmse"] = coi_norm
    row["actual_window_common_normalized_rmse"] = common_norm
    if not (row["finite_to_endpoint"] and row["full_fine_finite"] and row["full_coarse_finite"]
            and row["los_pair_pass"] and coi_norm <= LIMIT and common_norm <= LIMIT):
        raise ValueError(f"numerical/classification mismatch for {event['id']}")
    return _safe(row)


def physical_bus(event, historical_rows):
    """Use the original indexed fault_bus; discovery group is attested bus_id."""
    if event["root"] == "closure_discovery":
        bus = str(event["group"])
    else:
        row = historical_rows[event["root"]][event["id"]]
        bus = str(row["fault_bus"])
        if bus != str(event["group"]):
            raise ValueError("historical fault bus disagrees with train event")
    if not bus or bus.lower() in ("none", "nan"):
        raise ValueError("missing physical fault bus")
    return bus


def decide(rows):
    if len(rows) != 366 or any(r.get("numerical_pass") is not True for r in rows):
        return "INCONCLUSIVE_NUMERICAL_OR_IMPLEMENTATION", []
    chosen = [r for r in rows if r["N"] > 0 and all(
        r[step]["total"]["D"] >= 1 and r[step]["total"]["B"] >= 25 * EPSILON
        for step in ("fine", "coarse"))]
    status = ("ACTIVE_DESIGN_ADMISSIBLE" if len(chosen) >= 12 and
              len({r["fault_bus"] for r in chosen}) >= 3 else "NO_ACTIVE_DESIGN_ADMISSION")
    return status, chosen


def summary(rows, chosen):
    import numpy as np
    def stats(subset):
        out = {"count": len(subset)}
        for step in ("fine", "coarse"):
            for metric in ("B", "W", "D"):
                values = [r[step]["total"][metric] for r in subset if r.get(step)]
                out[f"{step}_{metric}"] = None if not values else {
                    "event_mean": float(np.mean(values)), "median": float(np.median(values)),
                    "min": float(np.min(values)), "max": float(np.max(values))}
        return out
    return {"all": stats(rows), "stable": stats([r for r in rows if r["label"] == "stable"]),
            "unstable": stats([r for r in rows if r["label"] == "unstable"]),
            "buses": {bus: stats([r for r in rows if r["fault_bus"] == bus])
                      for bus in sorted({r["fault_bus"] for r in rows})},
            "qualified_events": len(chosen), "qualified_buses": sorted({r["fault_bus"] for r in chosen})}


def infer(args, manifest):
    import csv
    import numpy as np
    import torch
    from closure_mechanism_study import train_cohort
    from fixed_edge_pair_20261019 import metadata
    from train_local_memory_0098 import graph
    from closure_training_runtime import TrainingClosureROM
    from linear_cache_runtime_20261019 import enable_linear_cache
    torch.set_num_threads(1)
    events, _, cohort = train_cohort()
    reference = metadata(events)
    if (len(events) != 366 or sum(e["label"] == "stable" for e in events) != 325 or
            sum(e["label"] == "unstable" for e in events) != 41 or
            len(cohort["old_excluded"]) + len(cohort["new_excluded"]) != 27):
        raise ValueError("registered 393/366 cohort or labels changed")
    expected_cohort = json.loads((P2 / "6101_full8/checkpoints/step001024/checkpoint.json").read_text())["identity"]["common"]["cohort"]
    if cohort != expected_cohort:
        raise ValueError("P2 frozen train cohort identity changed")
    historical = {}
    for root_name in ("ieee39_postfault", "ieee39_expansion"):
        with (ROOT / "private_data" / root_name / "dataset_index.csv").open(newline="", encoding="utf-8") as stream:
            historical[root_name] = {r["event_id"]: r for r in csv.DictReader(stream)}
    buses = [physical_bus(e, historical) for e in events]
    models, masses = {}, {}
    physical_invariants = {}
    for key in KEYS:
        path = P2 / key / "checkpoints/step001024/model.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        physical_invariants[key] = {"input_scale": payload["config"]["input_scale"],
                                    **{name: payload["physical"][name] for name in
                                       ("mass", "D", "delta_star", "phi", "graph_coupling",
                                        "input_scale_omega", "p_scale", "neighbor_A",
                                        "port_s_weight", "port_c_weight")}}
        model = enable_linear_cache(TrainingClosureROM.from_export(path).cpu()).double().eval()
        masses[key] = model.mass.detach().numpy().copy()
        models[key] = model
    mass = masses[KEYS[0]]
    if any(not np.array_equal(mass, m) for m in masses.values()):
        raise ValueError("six model inertia buffers differ")
    if any(physical_invariants[key] != physical_invariants[KEYS[0]] for key in KEYS):
        raise ValueError("six model fixed physical/normalization inputs differ")
    result = {"status": "INCONCLUSIVE_NUMERICAL_OR_IMPLEMENTATION", "split": "train",
              "raw_train_denominator": 393, "eligible_denominator": 366,
              "excluded": [{"source": source, **r} for source in ("old_excluded", "new_excluded")
                           for r in cohort[source]], "reference_fingerprint": reference["fingerprint_sha256"],
              "historical_science_commit": manifest["historical_science_commit"],
              "prospective_runner_sha256": manifest["prospective_runner_sha256"], "rows": []}
    for i, e in enumerate(events):
        n = min(500, len(e["x"]) - 11)
        result["rows"].append({"event_id": e["id"], "group": e["group"], "root": e["root"],
                               "fault_bus": buses[i], "label": e["label"], "N": n,
                               "prediction_start_record_s": .1,
                               "first_scored_record_s": .11 if n else None,
                               "last_scored_record_s": .1 + n*.01 if n else None,
                               "numerical_pass": None, "reason": "not_evaluated"})
    write(args.output / "partial.json", result)
    for first in range(0, 366, 32):
        if time.perf_counter() - args.started > args.max_seconds - 10:
            raise TimeoutError("registered P3 inference deadline")
        block = events[first:first+32]
        histories = np.stack([e["x"][:11] for e in block])
        h = torch.as_tensor(histories, dtype=torch.float64)
        predictions = {}
        for key, model in models.items():
            with torch.no_grad(), graph(model):
                fine = model.rollout(h, 1000, .005).detach().numpy()
                coarse = model.rollout(h, 500, .01).detach().numpy()
            if fine.shape != (len(block), 1001, 40) or coarse.shape != (len(block), 501, 40):
                raise ValueError(f"malformed rollout: {key}")
            predictions[key] = (fine, coarse)
        # Save complete physical predictions even when a following event is numerically unresolved.
        name = f"predictions_{first:03d}_{first+len(block)-1:03d}.npz"
        path = args.output / name
        np.savez_compressed(path, event_ids=np.array([e["id"] for e in block]),
                            **{key+"_"+step: array for key, pair in predictions.items()
                               for step, array in zip(("fine", "coarse"), pair)})
        batch_sha = sha(path)
        for i, e in enumerate(block):
            row = result["rows"][first+i]
            n = row["N"]
            row.update(prediction_file=name, prediction_file_sha256=batch_sha,
                       prediction_index=i, model_arrays={})
            if n <= 0:
                row.update(numerical_pass=False, reason="no prediction time")
                write(args.output / "partial.json", result)
                raise ValueError(f"zero prediction time: {e['id']}")
            try:
                for key, (f, c) in predictions.items():
                    row["model_arrays"][key] = {"fine_sha256": hashlib.sha256(np.ascontiguousarray(f[i]).tobytes()).hexdigest(),
                                                "coarse_sha256": hashlib.sha256(np.ascontiguousarray(c[i]).tobytes()).hexdigest()}
                    row.setdefault("numerical", {})[key] = numerical(f[i], c[i], histories[i], e,
                                                                      e["classification_meta"], mass, n)
                for step, index, stride in (("fine", 0, 2), ("coarse", 1, 1)):
                    row[step] = disagreement({key: predictions[key][index][i, stride:stride*(n+1):stride]
                                               for key in KEYS}, mass)
                row["numerical_pass"] = True
                row.pop("reason", None)
            except BaseException as exc:
                row.update(numerical_pass=False, reason=repr(exc))
                write(args.output / "partial.json", result)
                raise
        write(args.output / "partial.json", result)
    verify_manifest(manifest)
    status, chosen = decide(result["rows"])
    result.update(status=status, summary=summary(result["rows"], chosen))
    write(args.output / "final.json", result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("freeze", "qa", "infer"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--max-seconds", type=float, default=3600.)
    args = p.parse_args()
    if args.mode == "freeze":
        if args.output.exists():
            p.error("new manifest path required")
        write(args.output, frozen_inputs())
        return
    if args.output.exists():
        p.error("new output directory required")
    args.output.mkdir(parents=True)
    args.started = time.perf_counter()
    write(args.output / "worker.json", {"pid": os.getpid(), "argv": sys.argv,
                                        "started_unix": time.time(), "max_seconds": args.max_seconds})
    try:
        if args.mode == "qa":
            if not 0 < args.max_seconds <= 300:
                raise ValueError("QA cap is 300 seconds")
            sys.path.insert(0, str(ROOT / "tests"))
            from test_closure_p3_disagreement_20261020 import run_qa
            qa = run_qa()
            write(args.output / "qa.json", qa)
            if qa["status"] != "PASS":
                raise ValueError("synthetic QA failed")
        else:
            if not 0 < args.max_seconds <= 3600 or args.manifest is None:
                raise ValueError("inference requires frozen manifest and <=3600 seconds")
            manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
            verify_manifest(manifest)
            sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src")]
            infer(args, manifest)
    except BaseException as exc:
        write(args.output / "failure.json", {"status": "INCONCLUSIVE_NUMERICAL_OR_IMPLEMENTATION",
                                               "error": repr(exc), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
