from __future__ import annotations

from experiments.verify_artifacts import inspect_json_text, verify_json_artifacts


def test_json_gate_rejects_duplicate_keys() -> None:
    findings = inspect_json_text(
        '{"value": 1, "value": 2}',
        source="duplicate.json",
        strict_nonfinite=True,
    )
    assert [(item.level, item.message) for item in findings] == [
        ("error", "duplicate object key: 'value'")
    ]


def test_json_gate_distinguishes_strict_and_legacy_nonfinite() -> None:
    strict = inspect_json_text(
        '{"value": NaN}', source="nonfinite.json", strict_nonfinite=True
    )
    legacy = inspect_json_text(
        '{"value": NaN}', source="nonfinite.json", strict_nonfinite=False
    )
    assert strict[0].level == "error"
    assert legacy[0].level == "warning"


def test_repository_json_gate_passes() -> None:
    report = verify_json_artifacts()
    assert report["files_checked"] > 0
    assert report["status"] == "pass"
    assert report["errors"] == []
