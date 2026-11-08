"""Verify this batch and the cumulative archive, without model or data-payload access."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess


folder = Path(__file__).resolve().parent
root = folder.parents[2]
current = json.loads((folder / "preflight.json").read_text(encoding="utf-8"))
previous = json.loads((root / current["previous_batch"]).read_text(encoding="utf-8"))
all_moves = previous["moves"] + current["moves"]
locations = {entry["old_path"]: entry["new_path"] for entry in all_moves}
assert len(locations) == len(all_moves) == 22
assert len(current["moves"]) == 16
for entry in all_moves:
    data = (root / entry["new_path"]).read_bytes()
    assert len(data) == entry["bytes"], entry["new_path"]
    assert hashlib.sha256(data).hexdigest() == entry["sha256"], entry["new_path"]
    assert not (root / entry["old_path"]).exists(), entry["old_path"]
    blob = subprocess.check_output(["git", "show", current["source_commit"] + ":" + entry["old_path"]], cwd=root)
    assert data == blob, entry["old_path"]
for entry in current["protected"]:
    data = (root / entry["path"]).read_bytes()
    assert len(data) == entry["bytes"] and hashlib.sha256(data).hexdigest() == entry["sha256"]
for source in previous["sources"]:
    name = locations.get(source["original_path"], source["original_path"])
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == source["sha256"], name

baseline_file = root / "results/repository_organization/20261108_inventory01/inventory.jsonl"
baseline = [json.loads(line) for line in baseline_file.read_text(encoding="utf-8").splitlines()]
mutable_since_baseline = {"README.md", "reports/README.md", ".gitattributes"}
metadata_checks = text_checks = 0
for row in baseline:
    old = row["current_path"]
    if old in mutable_since_baseline:
        continue
    path = root / locations.get(old, old)
    info = path.stat()
    assert (info.st_size, info.st_mtime_ns) == (row["bytes"], row["mtime_ns"]), old
    metadata_checks += 1
    if row["sha256"]:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], old
        text_checks += 1

attr = subprocess.check_output(["git", "check-attr", "-z", "text", "--", *locations.values()], cwd=root).decode("utf-8").split("\0")
assert attr[2::3] == ["unset"] * 22
receipt = dict(current)
receipt.update(state="MOVED_AND_BYTE_VERIFIED", verified_at_utc=datetime.now(timezone.utc).isoformat(),
    cumulative_locations=[dict(old_path=entry["old_path"], current_path=entry["new_path"], sha256=entry["sha256"], bytes=entry["bytes"],
        source_record=current["previous_batch"] if entry in previous["moves"] else "results/repository_organization/20261108_early_history02/migration.json") for entry in all_moves],
    supersedes_path_protection=[dict(path=entry["path"], reason="Its frozen referring audit is archived in the same batch; fixed-commit source context and maintained readers are preserved.") for entry in previous["protected"]],
    prior_receipts_modified=False, maintained_files=[])
for entry in receipt["moves"]:
    entry.update(size_and_sha256_verified=True, source_git_blob_exact=True,
        old_path_absent=True, git_text_conversion="disabled", network_link_liveness_checked=False)
for entry in current["maintained_files_before"]:
    data = (root / entry["path"]).read_bytes()
    receipt["maintained_files"].append(dict(path=entry["path"], before_sha256=entry["sha256"], after_sha256=hashlib.sha256(data).hexdigest()))
(folder / "migration.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

readers = ["README.md", "reports/README.md", "reports/history/early_e_series/README.md",
           "reports/history/paper/README.md", "reports/maintenance/early_gates/README.md",
           "results/repository_organization/README.md"]
link = re.compile(r"!?\[[^\]]*\]\(([^\s)]+)(?:\s+[\"'][^)]*)?\)")
local_links = fixed_links = 0
for name in readers:
    text = (root / name).read_text(encoding="utf-8-sig")
    assert all(line.rstrip() == line for line in text.splitlines()), name
    for value in link.findall(text):
        if value.startswith(("https://", "http://")):
            if current["source_commit"] in value:
                old = value.split(current["source_commit"] + "/", 1)[1]
                assert old in locations, (name, value)
                fixed_links += 1
            continue
        if value.startswith("#"):
            continue
        path = ((root / name).parent / value.split("#", 1)[0]).resolve()
        assert path.is_relative_to(root) and path.exists(), (name, value)
        local_links += 1
assert fixed_links == 22
assert len(list((root / "reports/history/early_e_series/originals").glob("E[0-9]*.md"))) == 13
assert not list((root / "reports").glob("E[0-9]*.md"))
assert len(list((root / "reports/history/paper/originals").glob("*.md"))) == 7
assert len(list((root / "reports/maintenance/early_gates/originals").glob("*.md"))) == 1
result = dict(schema_version=1, moved_this_batch=16, moved_cumulative=22,
    early_e_reports_archived=13, retained_protected_paper_sources=3,
    source_hash_and_git_blob_checks=22, fixed_commit_reading_links_checked=fixed_links,
    local_reading_links_checked=local_links, baseline_files_metadata_checked=metadata_checks,
    baseline_text_hashes_checked=text_checks, unrelated_original_changes_observed=False,
    original_relative_links="kept verbatim; use fixed-commit source readers",
    network_link_liveness_checked=False, scientific_payloads_read=False, experiments_run=False)
(folder / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
