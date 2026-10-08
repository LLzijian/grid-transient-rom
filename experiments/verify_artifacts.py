#!/usr/bin/env python
"""Artifact validation gate (G6).

Public API
----------
inspect_json_text(text, *, source, strict_nonfinite) -> list[Finding]
    Structural inspection of one JSON text. ``Finding`` has ``.level`` in
    {"error", "warning"} and ``.message``.
verify_json_artifacts(root=None) -> dict
    Repository-wide scan of every ``*.json`` under ``results/`` and ``configs/``.
    Returns ``{"files_checked", "status", "errors", "warnings"}``.

Semantics
---------
* duplicate object keys            -> error (always)
* non-finite numbers (NaN/Infinity/-Infinity):
    ``strict_nonfinite=True``      -> error
    ``strict_nonfinite=False``     -> warning
* malformed JSON                   -> error

The repository scan runs in *legacy* mode (``strict_nonfinite=False``): only
errors fail the gate. The inherited scientific corpus legitimately contains
``Infinity`` (e.g. an infinite effective time constant recorded for a rejected
module candidate), so non-finite tokens are warnings there, not gate failures.

CLI
---
    python experiments/verify_artifacts.py --json-only
Exit code 0 iff the repository scan reports no errors, else 1.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = ("results", "configs")

_NONFINITE = {"NaN": float("nan"), "Infinity": float("inf"), "-Infinity": float("-inf")}


@dataclass(frozen=True)
class Finding:
    """One structural finding about a JSON text."""

    level: str  # "error" | "warning"
    message: str


def inspect_json_text(
    text: str, *, source: str, strict_nonfinite: bool
) -> list[Finding]:
    """Inspect a single JSON text.

    ``source`` is only used to name the artifact; findings do not embed it so
    callers can group findings per file themselves.
    """
    del source  # reserved for richer per-finding provenance
    issues: list[Finding] = []
    duplicates: list[str] = []

    def object_pairs_hook(pairs: list[tuple[str, object]]) -> dict:
        seen: set[str] = set()
        for key, _value in pairs:
            if key in seen:
                duplicates.append(key)
            else:
                seen.add(key)
        return dict(pairs)

    def parse_constant(token: str) -> float:
        level = "error" if strict_nonfinite else "warning"
        issues.append(Finding(level, f"non-finite number: {token}"))
        return _NONFINITE[token]

    try:
        json.JSONDecoder(
            object_pairs_hook=object_pairs_hook, parse_constant=parse_constant
        ).decode(text)
    except ValueError as exc:
        issues.append(Finding("error", f"malformed JSON: {exc}"))

    for key in duplicates:
        issues.append(Finding("error", f"duplicate object key: '{key}'"))
    return issues


def verify_json_artifacts(root: str | Path | None = None) -> dict:
    """Scan all registered JSON artifacts under the repository root."""
    repo_root = Path(root) if root is not None else ROOT
    files_checked = 0
    errors: list[str] = []
    warnings = 0
    for base in SCAN_DIRS:
        base_dir = repo_root / base
        if not base_dir.is_dir():
            continue
        for path in sorted(base_dir.rglob("*.json")):
            files_checked += 1
            rel = path.relative_to(repo_root)
            try:
                text = path.read_text(encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{rel}: unreadable ({exc!r})")
                continue
            for finding in inspect_json_text(
                text, source=str(rel), strict_nonfinite=False
            ):
                if finding.level == "error":
                    errors.append(f"{rel}: {finding.message}")
                else:
                    warnings += 1
    return {
        "files_checked": files_checked,
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Artifact validation gate (G6)."
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="validate registered JSON artifacts (the implemented check)",
    )
    parser.add_argument(
        "--repo-root",
        default=str(ROOT),
        help="repository root (default: parent of this file)",
    )
    args = parser.parse_args(argv)
    _ = args.json_only  # accepted for registry compatibility; sole implemented check

    report = verify_json_artifacts(args.repo_root)
    print(
        f"[verify_artifacts] json-only: scanned {report['files_checked']} *.json "
        f"under {', '.join(SCAN_DIRS)}; "
        f"warnings={report['warnings']}; status={report['status']}"
    )
    for error in report["errors"]:
        print("  -", error)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
