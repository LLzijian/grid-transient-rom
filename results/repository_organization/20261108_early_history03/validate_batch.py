"""Verify cumulative locations, protected sources, preserved bytes and declared readers."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess


folder = Path(__file__).resolve().parent
root = folder.parents[2]
plan = json.loads((folder / "preflight.json").read_text(encoding="utf-8"))
prior = json.loads((root / plan["previous_batch"]).read_text(encoding="utf-8"))
locations = {entry["old_path"]: dict(entry) for entry in prior["cumulative_locations"]}
for entry in plan["moves"]:
    assert entry["old_path"] not in locations
    locations[entry["old_path"]] = dict(old_path=entry["old_path"], current_path=entry["new_path"],
        bytes=entry["bytes"], sha256=entry["sha256"], git_blob_exact=entry["git_blob_exact"],
        git_text_equivalent_after_line_ending_normalization=True,
        source_record="results/repository_organization/20261108_early_history03/migration.json")
assert len(locations) == 181 and len(plan["moves"]) == 159 and len(plan["protected"]) == 23
exact_git = 0
for old, entry in locations.items():
    path = root / entry["current_path"]
    data = path.read_bytes()
    assert len(data) == entry["bytes"] and hashlib.sha256(data).hexdigest() == entry["sha256"], old
    assert not (root / old).exists(), old
    blob = subprocess.check_output(["git", "show", plan["source_commit"] + ":" + old], cwd=root)
    assert data.replace(b"\r\n", b"\n") == blob.replace(b"\r\n", b"\n"), old
    exact_git += data == blob
for entry in plan["protected"]:
    data = (root / entry["path"]).read_bytes()
    assert len(data) == entry["bytes"] and hashlib.sha256(data).hexdigest() == entry["sha256"], entry["path"]
    for edge in entry["evidence"]:
        assert (root / edge["source"]).exists(), edge["source"]

baseline = list(map(json.loads, (root / "results/repository_organization/20261108_inventory01/inventory.jsonl").read_text(encoding="utf-8").splitlines()))
metadata_checks = text_checks = 0
for row in baseline:
    old = row["current_path"]
    if old in {"README.md", "reports/README.md", ".gitattributes"}:
        continue
    current = locations.get(old, {}).get("current_path", old)
    path = root / current
    info = path.stat()
    assert (info.st_size, info.st_mtime_ns) == (row["bytes"], row["mtime_ns"]), old
    metadata_checks += 1
    if row["sha256"]:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], old
        text_checks += 1

attrs = subprocess.check_output(["git", "check-attr", "-z", "text", "--", *[entry["current_path"] for entry in locations.values()]], cwd=root).decode("utf-8").split("\0")
assert attrs[2::3] == ["unset"] * 181
receipt = dict(plan)
receipt.update(state="MOVED_AND_BYTE_VERIFIED", verified_at_utc=datetime.now(timezone.utc).isoformat(),
    cumulative_locations=list(locations.values()), prior_receipts_modified=False, maintained_files=[])
for entry in receipt["moves"]:
    entry.update(size_and_sha256_verified=True, old_path_absent=True, git_text_conversion="disabled", network_link_liveness_checked=False)
for entry in plan["maintained_files_before"]:
    data = (root / entry["path"]).read_bytes()
    receipt["maintained_files"].append(dict(path=entry["path"], before_sha256=entry["sha256"], after_sha256=hashlib.sha256(data).hexdigest()))
receipt["released_prior_protection"] = [dict(path=entry["path"], reason="Its citing 0004 acceptance is included in this closed reference group; all maintained readers and fixed-commit contexts are preserved.")
    for entry in prior["protected"] if entry["path"] in locations]
(folder / "migration.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

readers = ["README.md", "reports/README.md", "reports/history/early_e_series/README.md",
    "reports/history/early_e_series/catalog.md", "reports/history/paper/README.md", "reports/history/paper/catalog.md",
    "reports/maintenance/early_gates/README.md", "reports/maintenance/early_gates/decision_readers.md",
    "reports/history/private_rom/README.md", "reports/history/private_rom/catalog.md",
    "reports/history/private_rom/protected.md", "results/repository_organization/README.md"]
links = re.compile(r"!?\[[^\]]*\]\(([^\s)]+)(?:\s+[\"'][^)]*)?\)")
local_links = fixed_links = 0
known_old = set(locations) | {entry["path"] for entry in plan["protected"]}
for name in readers:
    path = root / name
    text = path.read_text(encoding="utf-8-sig")
    assert all(line.rstrip() == line for line in text.splitlines()), name
    for value in links.findall(text):
        if value.startswith(("https://", "http://")):
            if plan["source_commit"] in value:
                old = value.split(plan["source_commit"] + "/", 1)[1]
                assert old in known_old, (name, value)
                fixed_links += 1
            continue
        if value.startswith("#"):
            continue
        target = (path.parent / value.split("#", 1)[0]).resolve()
        assert target.is_relative_to(root) and target.exists(), (name, value)
        local_links += 1
assert len(list((root / "reports/history/early_e_series/originals").glob("E[0-9]*.md"))) == 13
assert len(list((root / "reports/history/private_rom/originals").rglob("*.md"))) == 119
assert sum(entry["old_path"].endswith(".json") for entry in plan["moves"]) == 21
result = dict(schema_version=1, candidates=182, moved_this_batch=159, moved_cumulative=181,
    moved_markdown_this_batch=138, moved_machine_evidence_this_batch=21, protected_originals_unchanged=23,
    baseline_files_metadata_checked=metadata_checks, baseline_text_hashes_checked=text_checks,
    original_working_tree_byte_hashes_preserved=181, git_blob_exact=exact_git,
    git_differs_only_by_line_endings=181-exact_git, local_reading_links_checked=local_links,
    fixed_commit_reader_links_checked=fixed_links, current_raw_datasets_models_code_unchanged=True,
    numerical_evidence_structure_precheck=plan["json_structural_precheck"],
    raw_trajectories_or_weights_parsed=False, experiments_run=False, network_link_liveness_checked=False)
(folder / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
