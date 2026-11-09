"""Inventory the closed precision Run's raw files and verify its public references."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re


ROOT = Path(__file__).resolve().parents[2]
RUN = "gftd_initialization_precision_20261109"
RAW = Path("artifacts") / RUN
RESULTS = ROOT / "results" / RUN
TERMINAL = RESULTS / "candidate/full/execution.json"
OUTPUT = RESULTS / "raw_artifacts_manifest.json"
HEX_SHA = re.compile(r"[a-fA-F0-9]{64}")


def raw_reference(value):
    if not isinstance(value, str):
        return None
    path = PureWindowsPath(value)
    if path.is_absolute():
        try:
            path = path.relative_to(PureWindowsPath(ROOT))
        except ValueError:
            return None
    try:
        tail = path.relative_to(PureWindowsPath(RAW))
    except ValueError:
        return None
    return (PureWindowsPath(RAW) / tail).as_posix()


def scan_references(value, pairs, paths):
    if isinstance(value, list):
        for item in value:
            scan_references(item, pairs, paths)
    elif isinstance(value, dict):
        for key, item in value.items():
            mapped = raw_reference(key)
            if mapped is not None and isinstance(item, str):
                pairs[(mapped, item.lower())].add(None)
            direct = raw_reference(item)
            if direct is not None:
                paths.add(direct)
                if key == "path":
                    expected, size = value.get("sha256"), value.get("bytes")
                else:
                    expected, size = value.get(f"{key.removesuffix('_path')}_sha256"), None
                if isinstance(expected, str):
                    pairs[(direct, expected.lower())].add(size if type(size) is int else None)
            scan_references(item, pairs, paths)


def main():
    terminal = TERMINAL.read_bytes()
    if json.loads(terminal).get("status") != "COMPLETED":
        raise SystemExit("full candidate execution must be COMPLETED before raw inventory")
    if OUTPUT.exists():
        raise SystemExit(f"manifest already exists: {OUTPUT}")

    raw_root = ROOT / RAW
    if raw_root.resolve() != raw_root or not raw_root.is_dir():
        raise ValueError(f"unexpected raw directory: {raw_root}")
    files, directories, categories = {}, {RAW.as_posix()}, {}
    for path in sorted(raw_root.rglob("*")):
        if path.resolve() != path:
            raise ValueError(f"unexpected link in raw directory: {path}")
        relative = path.relative_to(ROOT).as_posix()
        if path.is_dir():
            directories.add(relative)
            continue
        if not path.is_file():
            raise ValueError(f"unexpected raw directory entry: {path}")
        before = path.stat()
        with path.open("rb") as stream:
            sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"raw file changed during hashing: {path}")
        subpath = path.relative_to(raw_root)
        category = subpath.parts[0] if len(subpath.parts) > 1 else "root"
        files[relative] = dict(path=relative, bytes=after.st_size, sha256=sha256,
                               category=category, local_only=True, non_upload=True)
        counts = categories.setdefault(category, dict(files=0, bytes=0))
        counts["files"] += 1
        counts["bytes"] += after.st_size

    pairs, paths, json_count = defaultdict(set), set(), 0
    for path in sorted(RESULTS.rglob("*.json")):
        if path == OUTPUT:
            continue
        if path.resolve() != path:
            raise ValueError(f"unexpected result link: {path}")
        scan_references(json.loads(path.read_bytes()), pairs, paths)
        json_count += 1
    mismatches = []
    for (relative, expected), sizes in sorted(pairs.items()):
        actual = files.get(relative)
        stated_sizes = sorted(size for size in sizes if size is not None)
        if (actual is None or HEX_SHA.fullmatch(expected) is None or actual["sha256"] != expected
                or any(size != actual["bytes"] for size in stated_sizes)):
            mismatches.append(dict(path=relative, expected_sha256=expected, expected_bytes=stated_sizes,
                                   actual_sha256=None if actual is None else actual["sha256"],
                                   actual_bytes=None if actual is None else actual["bytes"]))
    path_only = paths - {path for path, _ in pairs}
    if TERMINAL.read_bytes() != terminal:
        raise ValueError("terminal candidate receipt changed during inventory")
    verified = bool(files and pairs and not mismatches)
    manifest = dict(
        schema="gftd_initialization_precision_raw_artifacts_manifest_v1",
        recorded_utc=datetime.now(timezone.utc).isoformat(),
        primary_root=RAW.as_posix(), publication="local_private_not_uploaded", local_only=True, non_upload=True,
        integrity_status="verified" if verified else "unresolved",
        full_execution_sha256=hashlib.sha256(terminal).hexdigest(),
        file_count=len(files), total_primary_bytes=sum(record["bytes"] for record in files.values()),
        result_json_scanned=json_count, unique_path_sha256_references=len(pairs),
        verified_references=len(pairs) - len(mismatches), mismatched_references=mismatches,
        path_only_reference_count=len(path_only), path_only_missing=sorted(path_only - files.keys() - directories),
        category_counts=categories, files=list(files.values()),
    )
    with OUTPUT.open("xb") as stream:
        stream.write((json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(f"{manifest['integrity_status']}: {len(files)} files, {manifest['total_primary_bytes']} bytes, "
          f"{manifest['verified_references']}/{len(pairs)} hashed references")
    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
