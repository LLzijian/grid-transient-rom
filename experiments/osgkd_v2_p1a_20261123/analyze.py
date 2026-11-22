"""Audit the two fixed P1a endpoints from their saved scores and local receipts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import median
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from kiki.evaluation.osgkd import summarize

RUN = "osgkd_v2_p1a_20261123"
PARENT = "osgkd_fit297_20261113"
RESULTS = ROOT / "results" / RUN
LOCAL = ROOT / "artifacts" / RUN / "pair01"


def read(path):
    return json.loads(path.read_bytes())


def identity(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        size = stream.tell()
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": digest, "bytes": size}


def verified(ref):
    path = (ROOT / ref["path"]).resolve()
    if not path.is_relative_to(ROOT) or identity(path) != ref:
        raise ValueError(f"frozen identity differs for {ref['path']}")
    return path


def event_changes(reference, model):
    old, new = set(reference), set(model)
    return {"retained": sorted(old & new), "added": sorted(new - old), "removed": sorted(old - new)}


def timing(rows):
    common = [
        row["event_id"]
        for row in rows["parent"].values()
        if row["label"] == "unstable"
        and all(model[row["event_id"]]["classification_actual"]["timing_comparable"]
                for model in rows.values())
    ]
    delays = {}
    for name, model in rows.items():
        delays[name] = {
            event_id: model[event_id]["classification_actual"]["trigger_time_error_s"]
            for event_id in common
        }
    for event_id in common:
        times = {model[event_id]["classification_actual"]["true_time_record_s"] for model in rows.values()}
        assert len(times) == 1
    return {"common_ids": common, "delay_s_by_id": delays}


def summarize_attempts(attempts):
    return {
        "attempts": len(attempts),
        "accepted": sum(row["accepted"] for row in attempts),
        "clipped": sum(row["clipped"] for row in attempts),
        "gradient_norm_median": median(row["gradient_norm"] for row in attempts if row["accepted"]),
        "loss_medians_by_50": [
            median(row["J"] for row in attempts[start:start + 50] if row["accepted"])
            for start in range(0, len(attempts), 50)
        ],
    }


def main():
    assert not (LOCAL / "failed.json").exists()
    completion = read(LOCAL / "completed.json")
    summary_path = verified(completion["summary_ref"])
    receipt = read(summary_path)
    release = read(verified(receipt["release_ref"]))
    assert release == receipt["release"]
    assert receipt["status"] == completion["status"] == "completed" and receipt["phase"] == "compare"
    assert receipt["run_id"] == RUN and receipt["job_id"] == "pair01"
    result = receipt["result"]
    assert result["fixed_endpoint"] == 1700 and result["evaluation_denominator"] == 297
    for name, digest in release["source_hashes"].items():
        assert identity(ROOT / name)["sha256"] == digest

    parent_results = ROOT / "results" / PARENT
    material = read(verified(read(verified(release["audit_ref"]))["materialization"]))
    parent_manifest = read(parent_results / "artifacts.json")
    assert parent_manifest["selected_models"]["C1_dynamic"] == release["checkpoint_ref"]
    verified(release["checkpoint_ref"])
    parent_path = ROOT / "artifacts" / PARENT / "eval01" / "C1_dynamic"
    files = {ref["path"]: ref for ref in parent_manifest["files"]}
    paths = {"parent": parent_path, **{name: LOCAL / name / "evaluation" for name in ("2s", "8s")}}
    saved, rows, artifact_refs = {}, {}, []
    for name, directory in paths.items():
        if name == "parent":
            for basename in ("rows.json", "summary.json"):
                verified(files[(directory / basename).relative_to(ROOT).as_posix()])
            provenance = read(parent_results / "eval01" / "summary.json")
            official = provenance["result"]["scores"]["C1_dynamic"]
        else:
            treatment = result["treatments"][name]
            assert verified(treatment["evaluation_summary_ref"]) == directory / "summary.json"
            official = read(directory / "summary.json")
            artifact_refs.append(identity(verified(treatment["checkpoint"])))
            artifact_refs.append(identity(directory / "predictions.npz"))

        row_list = read(directory / "rows.json")
        assert [row["event_id"] for row in row_list] == material["ordered_ids"]
        assert len(set(material["ordered_ids"])) == 297
        assert sum(row["label"] == "unstable" for row in row_list) == 30
        recomputed = summarize(row_list, material)
        assert recomputed == {key: official[key] for key in recomputed}
        saved[name] = recomputed
        rows[name] = {row["event_id"]: row for row in row_list}
        artifact_refs.extend(identity(directory / basename) for basename in ("rows.json", "summary.json"))

    sequence = read(verified(result["paired_batch_sequence_ref"]))
    assert len(sequence) == 200
    train, gates = {}, {}
    for name in ("2s", "8s"):
        arm = LOCAL / name
        attempts = [json.loads(line) for line in (arm / "attempts.jsonl").read_bytes().splitlines()]
        assert attempts == read(arm / "attempts.json")
        assert [row["attempt"] for row in attempts] == list(range(1501, 1701))
        assert [row["event_ids"] for row in attempts] == sequence
        assert all(row["horizon_s"] == int(name[0]) for row in attempts)
        train[name] = summarize_attempts(attempts)
        assert all(train[name][key] == result["treatments"][name][key]
                   for key in ("attempts", "accepted", "clipped"))
        artifact_refs.extend(identity(arm / basename) for basename in ("attempts.json", "attempts.jsonl"))
        all_rows = rows[name].values()
        numerical = [row["scopes"]["8"]["numerical"] for row in all_rows]
        classes = [row["classification_8s"] for row in all_rows]
        gates[name] = {
            "qualified_8s": sum(row["passed"] for row in numerical),
            "max_normalized_rms": max(row["rms"] for row in numerical if row["rms"] is not None),
            "max_normalized_difference": max(row["maximum"] for row in numerical if row["maximum"] is not None),
            "los_pair_pass_8s": sum(row["los_pair_pass"] for row in classes),
            "settled_disagreements_8s": sum(row["settled_disagreement"] for row in classes),
        }

    parent = saved["parent"]["classification"]["train_all"]
    changes = {}
    for name in ("2s", "8s"):
        candidate = saved[name]["classification"]["train_all"]
        changes[name] = {
            "actual_detection_from_parent": event_changes(
                parent["actual"]["detected_unstable_ids"], candidate["actual"]["detected_unstable_ids"]),
            "stable_false_alarm_from_parent": event_changes(
                parent["8s"]["false_los_ids"], candidate["8s"]["false_los_ids"]),
        }

    stable = [event_id for event_id, row in rows["parent"].items() if row["label"] == "stable"]
    risks = [saved[name]["splits"]["train_all"]["stable_8s"]["official"]["motion_risk_v1"]
             for name in ("2s", "8s")]
    differences = [
        rows["8s"][event_id]["scopes"]["8"]["metrics"]["mean_motion_error"]
        - rows["2s"][event_id]["scopes"]["8"]["metrics"]["mean_motion_error"]
        for event_id in stable
    ]
    payload = {
        "run_id": RUN,
        "method": "recomputed official aggregates from saved rows; no model or trajectory replay",
        "source": identity(Path(__file__).resolve()),
        "endpoint_receipt": completion["summary_ref"],
        "artifacts": artifact_refs,
        "study": {
            "stable_8s_risk_long_minus_short": risks[1] - risks[0],
            "stable_8s_relative_risk_reduction": 1 - risks[1] / risks[0],
            "stable_8s_event_error_lower_count": sum(value < 0 for value in differences),
        },
        "scores": saved,
        "numerical": gates,
        "training": train,
        "changed_events": changes,
        "timing": timing(rows),
    }
    with (RESULTS / "analysis.json").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
