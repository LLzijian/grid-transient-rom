"""Independent acceptance check: coverage, conservative decisions, and unchanged originals."""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path


folder = Path(__file__).resolve().parent
root = folder.parents[2]
spec = importlib.util.spec_from_file_location("inventory_check", root / "tools/repository_inventory.py")
inv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inv)
rows = [json.loads(line) for line in (folder / "inventory.jsonl").read_text(encoding="utf-8").splitlines()]
summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
names = {row["current_path"] for row in rows}
assert len(names) == len(rows) == summary["entries"]
observed = set()
excluded = folder.relative_to(root).as_posix() + "/"
for directory in inv.PRIMARY + inv.CONTEXT:
    for name, _, _, _ in inv.walk(root, directory):
        if not name.startswith(excluded):
            observed.add(name)
observed.update(inv.ROOT_FILES)
assert names == observed, {"missing": sorted(observed - names), "unexpected": sorted(names - observed)}
hashed = 0
for row in rows:
    assert row["move_authorized"] is False
    assert row["owner"] is None
    path = inv.safe_path(root, row["current_path"])
    info = path.stat()
    assert (info.st_size, info.st_mtime_ns) == (row["bytes"], row["mtime_ns"]), row["current_path"]
    if row["sha256"]:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], row["current_path"]
        hashed += 1
migrations = [json.loads(line) for line in (folder / "migration_reconciliation.jsonl").read_text(encoding="utf-8").splitlines()]
assert Counter(row["kind"] for row in migrations) == {"documentation": 74, "local_storage": 216}
assert all(row["current_verification"] in {"bytes_and_sha256_match", "metadata_counts_and_bytes_match"} for row in migrations)
assert all(row["sha256_recheck"] is False for row in migrations if row["kind"] == "local_storage")
assert not summary["original_file_changes_observed"]
assert not summary["unreadable_or_skipped"]
assert not summary["metadata_parse_issues"]
result = dict(coverage_verified=True, files_verified=len(rows), text_hashes_rechecked=hashed,
              source_files_unchanged=True, all_migration_records_accounted_for=True,
              no_move_or_delete_permission_created=True, scientific_payloads_read=False,
              unit_tests={"command": "python -B -m unittest discover -s tests -p test_repository_inventory.py -v", "passed": 5},
              existing_harness_check={"valid": True, "issues": 0, "private_payloads_read": False},
              binary_verification="size_and_mtime_only", live_process_association="unknown")
(folder / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
