"""Verify the archive sample without reading scientific payloads or executing models."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess


folder = Path(__file__).resolve().parent
root = folder.parents[2]
preflight = json.loads((folder / "preflight.json").read_text(encoding="utf-8"))
changes = {entry["old_path"]: entry["new_path"] for entry in preflight["moves"]}
assert len(changes) == 6 and len(preflight["protected"]) == 7
for source in preflight["sources"]:
    old = source["original_path"]
    current = changes.get(old, old)
    data = (root / current).read_bytes()
    assert len(data) == source["bytes"], current
    assert hashlib.sha256(data).hexdigest() == source["sha256"], current
    if old in changes:
        assert not (root / old).exists(), old
        blob = subprocess.check_output(["git", "show", preflight["source_commit"] + ":" + old], cwd=root)
        assert data == blob, old

# Verify the old inventory as a historical baseline, allowing only this batch's declared changes.
old_inventory = root / "results/repository_organization/20261108_inventory01/inventory.jsonl"
baseline = [json.loads(line) for line in old_inventory.read_text(encoding="utf-8").splitlines()]
allowed_edit = {"README.md", "reports/README.md", ".gitattributes"}
unmodified, text_hashes = 0, 0
for row in baseline:
    name = row["current_path"]
    if name in allowed_edit or name in changes:
        continue
    info = (root / name).stat()
    assert (info.st_size, info.st_mtime_ns) == (row["bytes"], row["mtime_ns"]), name
    if row["sha256"]:
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == row["sha256"], name
        text_hashes += 1
    unmodified += 1

receipt = dict(preflight)
receipt["state"] = "MOVED_AND_BYTE_VERIFIED"
receipt["verified_at_utc"] = datetime.now(timezone.utc).isoformat()
for source in receipt["sources"]:
    source["current_path"] = changes.get(source["original_path"], source["original_path"])
for entry in receipt["moves"]:
    entry.update(size_and_sha256_verified=True, old_path_absent=True, local_git_reading_source_verified=True,
                 network_link_liveness_checked=False)
receipt["post_preflight_scope_addition"] = {"path": ".gitattributes", "reason": "Preserve exact bytes across future Windows checkouts, scoped only to the new originals directory."}
receipt["maintained_files"] = []
for name in sorted(allowed_edit):
    before = subprocess.check_output(["git", "show", preflight["source_commit"] + ":" + name], cwd=root)
    after = (root / name).read_bytes()
    receipt["maintained_files"].append(dict(path=name, before_sha256=hashlib.sha256(before).hexdigest(),
        after_sha256=hashlib.sha256(after).hexdigest(), operation="archive_line_endings_preserved" if name == ".gitattributes" else "navigation_updated"))
attributes = subprocess.check_output(["git", "check-attr", "-z", "text", "--", *changes.values()], cwd=root).decode("utf-8").split("\0")
assert attributes[2::3] == ["unset"] * 6
(folder / "migration.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

link_pattern = re.compile(r"!?\[[^\]]*\]\(([^\s)]+)(?:\s+[\"'][^)]*)?\)")
readers = ["README.md", "reports/README.md", "reports/history/early_e_series/README.md"]
local_links, fixed_links = 0, 0
for name in readers:
    text = (root / name).read_text(encoding="utf-8-sig")
    for value in link_pattern.findall(text):
        if value.startswith(("https://", "http://")):
            if preflight["source_commit"] in value:
                old = value.split(preflight["source_commit"] + "/", 1)[1]
                assert old in changes
                fixed_links += 1
            continue
        if value.startswith("#"):
            continue
        path = ((root / name).parent / value.split("#", 1)[0]).resolve()
        assert path.is_relative_to(root), (name, value)
        assert path.exists(), (name, value)
        local_links += 1
assert fixed_links == 6

new_inventory = list((root / "reports").glob("E[0-9]*.md"))
archived = list((root / "reports/history/early_e_series/originals").glob("E[0-9]*.md"))
assert len(new_inventory) == 7 and len(archived) == 6
result = dict(schema_version=1, source_files_verified=len(preflight["sources"]),
    moved_files=6, protected_originals_unchanged=7, e_reports_total=13,
    other_baseline_files_unchanged_by_metadata=unmodified, other_baseline_text_hashes_verified=text_hashes,
    reading_entry_local_links_checked=local_links, fixed_commit_sources_verified=fixed_links,
    archived_originals_git_text_attribute="unset for all six; byte preservation survives checkout",
    archived_original_relative_links="kept verbatim; use fixed-commit reading URLs for original link context",
    scientific_arrays_or_weights_read=False, experiments_run=False,
    historical_inventory_rewritten=False, new_scientific_conclusions_or_promotion=False)
(folder / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
