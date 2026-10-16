"""Synthetic-only tests for mandatory A TRAIN conditions in D's fixed selection."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

STAGING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGING / "experiments"))
import closure_discovery_data as data
import closure_discovery_confirmation as confirmation


def _condition(identifier, bus, line, duration, xf, *, event_id=None):
    return {
        "id": identifier,
        "event_id": event_id or identifier,
        "bus": bus,
        "bus_id": str(bus),
        "line": line,
        "location": 1.0,
        "duration_s": duration,
        "xf": xf,
        "split": "train",
    }


def _old_train_rows():
    return [
        _condition("old17", 17, "Line_21", 0.1, 0.01),
        _condition("old8", 8, "Line_10", 0.1, 0.01),
    ]


def _a_conditions():
    return [
        _condition("449f2a97fdcc3c2a", 17, "Line_21", 0.125, 0.01,
                   event_id="newtrain_449f2a97fdcc3c2a"),
        _condition("83adbaea1acb2425", 8, "Line_10", 0.175, 0.0031622776601683794,
                   event_id="newtrain_83adbaea1acb2425"),
    ]


def _pool_with_mandatory(a_conditions):
    pool = list(a_conditions)
    for stratum, duration, bus, line in (
        ("short", 0.125, 17, "Line_21"),
        ("medium", 0.175, 8, "Line_10"),
        ("long", 0.35, 17, "Line_21"),
    ):
        for index in range(8):
            pool.append(_condition(
                f"{stratum}-{index}", bus, line, duration,
                0.02 + 0.001 * index + (0.0001 if stratum == "long" else 0.0),
            ))
    return pool


def test_mandatory_conditions_are_included_and_other_slots_follow_disagreement(monkeypatch):
    old = _old_train_rows()
    a_conditions = _a_conditions()
    pool = _pool_with_mandatory(a_conditions)
    monkeypatch.setattr(data, "allowed_train_conditions", lambda: old)

    selected = data.select_new_train_conditions(
        pool,
        {"old17": 1.0, "old8": 2.0},
        count=24,
        mandatory_conditions=a_conditions,
    )

    keys = {data._physical_key(row) for row in selected}
    mandatory_keys = {data._physical_key(row) for row in a_conditions}
    assert mandatory_keys <= keys
    assert len(selected) == 24 and len(keys) == 24
    assert {name: sum(row["stratum"] == name for row in selected)
            for name in ("short", "medium", "long")} == {
                "short": 8, "medium": 8, "long": 8,
            }
    assert all("selection_score" in row for row in selected
               if data._physical_key(row) not in mandatory_keys)


def test_mandatory_conditions_must_be_distinct_and_present_in_pool(monkeypatch):
    old = _old_train_rows()
    monkeypatch.setattr(data, "allowed_train_conditions", lambda: old)
    a_conditions = _a_conditions()
    pool = _pool_with_mandatory(a_conditions)

    with pytest.raises(ValueError, match="duplicate mandatory"):
        data.select_new_train_conditions(
            pool, {"old17": 1.0, "old8": 2.0}, count=24,
            mandatory_conditions=[a_conditions[0], a_conditions[0]],
        )
    with pytest.raises(ValueError, match="absent from the distinct candidate pool"):
        data.select_new_train_conditions(
            pool[2:], {"old17": 1.0, "old8": 2.0}, count=24,
            mandatory_conditions=a_conditions,
        )


def test_existing_selection_omitting_mandatory_a_conditions_is_rejected(tmp_path, monkeypatch):
    a_conditions = _a_conditions()
    pool = _pool_with_mandatory(a_conditions)
    # An extra non-mandatory candidate in each affected stratum allows a complete
    # 8/8/8 selection that is nevertheless invalid because it omits A's conditions.
    pool.extend([
        _condition("short-extra", 17, "Line_21", 0.125, 0.031),
        _condition("medium-extra", 8, "Line_10", 0.175, 0.032),
    ])
    manifest = {"train_pool": pool}
    old = _old_train_rows()
    monkeypatch.setattr(data, "allowed_train_conditions", lambda: old)
    monkeypatch.setattr(
        data,
        "_conditions_for_split",
        lambda rows, split: [{**row, "split": split} for row in rows],
    )

    out = tmp_path
    generation_path = out / "A/fom_cost_retry2/generation.json"
    generation_path.parent.mkdir(parents=True)
    generation_path.write_text(json.dumps({
        "split": "train", "status": "complete", "conditions": a_conditions,
    }), encoding="utf-8")
    mandatory, generation_sha = confirmation._mandatory_a_train_conditions(out, data)
    base_identity = confirmation._selection_identity(
        manifest, [], [], mandatory, generation_sha,
    )

    bad_selection = []
    for stratum in ("short", "medium", "long"):
        matches = [row for row in pool if data._condition_stratum(row) == stratum
                   and data._physical_key(row) not in {data._physical_key(a) for a in a_conditions}]
        bad_selection.extend(matches[:8])
    selection_path = out / "D/selected_new_train.json"
    selection_path.parent.mkdir(parents=True)
    selection_path.write_text(json.dumps({
        "identity": {
            "base": base_identity,
            "conditions_sha256": confirmation._json_sha256(bad_selection),
        },
        "conditions": bad_selection,
    }), encoding="utf-8")

    assert base_identity["mandatory_a_train_conditions_sha256"] == confirmation._json_sha256(mandatory)
    assert base_identity["mandatory_a_train_generation_sha256"] == hashlib.sha256(
        generation_path.read_bytes()).hexdigest()
    with pytest.raises(confirmation.RecoveryUnresolved, match="omits a mandatory A TRAIN condition"):
        confirmation._select_or_reuse_train_conditions(
            out, manifest, [], [], None, None,
        )


def test_mandatory_a_manifest_requires_train_only_two_conditions(tmp_path):
    import closure_discovery_data as local_data

    generation_path = tmp_path / "A/fom_cost_retry2/generation.json"
    generation_path.parent.mkdir(parents=True)
    generation_path.write_text(json.dumps({
        "split": "val", "status": "complete", "conditions": _a_conditions(),
    }), encoding="utf-8")
    with pytest.raises(confirmation.RecoveryUnresolved, match="TRAIN-only"):
        confirmation._mandatory_a_train_conditions(tmp_path, local_data)


def test_selection_identity_binds_each_model_name_and_seed(tmp_path):
    rows = []
    for name, seed, contents in (
        ("main", 2701, b"main-2701"),
        ("main", 2702, b"main-2702"),
        ("alternative", 2701, b"alternative-2701"),
    ):
        path = tmp_path / f"{name}-{seed}.json"
        path.write_bytes(contents)
        rows.append({
            "name": name,
            "seed": seed,
            "path": str(path),
            "sha256": hashlib.sha256(contents).hexdigest(),
        })
    manifest = {"train_pool": []}
    identity = confirmation._selection_identity(manifest, rows, [], [], "a" * 64)
    assert set(identity["source_model_sha256"]) == {
        "main:2701", "main:2702", "alternative:2701",
    }

    changed_path = Path(rows[0]["path"])
    changed_path.write_bytes(b"main-2701-updated")
    rows[0]["sha256"] = hashlib.sha256(changed_path.read_bytes()).hexdigest()
    updated = confirmation._selection_identity(manifest, rows, [], [], "a" * 64)
    assert updated["source_model_sha256"]["main:2701"] != identity["source_model_sha256"]["main:2701"]

    with pytest.raises(confirmation.RecoveryUnresolved, match="duplicate frozen model identity"):
        confirmation._selection_identity(manifest, rows + [rows[0]], [], [], "a" * 64)
