"""Prepare this bounded archive batch; fail before moving any file."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


out = Path(__file__).resolve().parent
root = out.parents[2]
assert not (out / "preflight.json").exists(), "preflight already exists"
plan = json.loads((root / ".tmp/early_archive_plan03.json").read_text(encoding="utf-8"))
prior_path = "results/repository_organization/20261108_early_history02/migration.json"
prior = json.loads((root / prior_path).read_text(encoding="utf-8"))
rows = {r["current_path"]: r for r in map(json.loads, (root / "results/repository_organization/20261108_inventory01/inventory.jsonl").read_text(encoding="utf-8").splitlines())}
head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
sys.path.insert(0, str(root / "experiments"))
import verify_artifacts

paper = {"reports/evidence_governance_0003_manuscript_v2.md", "reports/manuscript_screen_rewrite_plan_v1.md",
         "reports/history/manuscript_rewrite_inflight_verify_v1.md", "reports/accepted/0004_manuscript_honesty_rescope.md",
         "reports/accepted/0005_manuscript_scientific_consistency.md"}


def destination(name):
    path = Path(name)
    match = re.match(r"(\d{4})_", path.name)
    number = int(match.group(1)) if match else None
    category = path.parent.name if path.parent.name in ("accepted", "failed") else "protocols"
    if path.suffix == ".json":
        group = "early_e_series_evidence" if number is not None and number <= 16 else "private_rom_history"
        return f"results/{group}/{category}/{path.name}", "machine_evidence"
    if name in paper:
        return f'reports/history/paper/originals/{category if match else "notes"}/{path.name}', "paper"
    if number == 0:
        return f"reports/maintenance/early_gates/originals/{category}/{path.name}", "maintenance"
    group = "early_e_series" if number is not None and number <= 16 else "private_rom"
    return f"reports/history/{group}/originals/{category}/{path.name}", group


moves, sources = [], []
json_warnings = 0
for name in plan["candidates"]:
    data = (root / name).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    assert digest == rows[name]["sha256"], name
    blob = subprocess.check_output(["git", "show", head + ":" + name], cwd=root)
    assert data.replace(b"\r\n", b"\n") == blob.replace(b"\r\n", b"\n"), name
    identity = dict(original_path=name, current_path=name, bytes=len(data), sha256=digest,
                    git_blob_exact=data == blob, git_blob_sha256=hashlib.sha256(blob).hexdigest(),
                    git_text_equivalent_after_line_ending_normalization=True)
    sources.append(identity)
    if name not in plan["eligible"]:
        continue
    new, group = destination(name)
    assert not (root / new).exists(), new
    if name.endswith(".json"):
        findings = verify_artifacts.inspect_json_text(data.decode("utf-8-sig"), source=name, strict_nonfinite=False)
        assert not [finding for finding in findings if finding.level == "error"], name
        json_warnings += sum(finding.level == "warning" for finding in findings)
    moves.append(dict(identity, old_path=name, new_path=new, group=group, operation="move_byte_identical",
        original_category=Path(name).parent.name, reading_url="https://github.com/LLzijian/grid-transient-rom/blob/" + head + "/" + name,
        relative_links="Frozen originals retain the old location context; use the fixed-commit reader. Some working-tree originals differ from Git only in CRLF/LF.",
        rollback="Restore verified new_path to absent old_path; restore maintained readers from preflight before-images."))
assert len(moves) == 159 and sum(entry["old_path"].endswith(".json") for entry in moves) == 21
assert len({entry["new_path"] for entry in moves}) == len(moves)
mutable = ["reports/README.md", "reports/history/early_e_series/README.md", "reports/history/paper/README.md",
           "reports/maintenance/early_gates/README.md", "results/repository_organization/README.md", ".gitattributes"]
maintained = []
for name in mutable:
    data = (root / name).read_bytes()
    maintained.append(dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), content_utf8=data.decode("utf-8")))
protected = []
for name, refs in sorted(plan["protected"].items()):
    if any(entry["source"].endswith((".py", ".ps1")) for entry in refs):
        reason = "脚本直接读取，原路径属于可执行依赖"
        condition = "调用者在获准的新版本中改址并验证输入身份；保留旧执行恢复方式，不改旧哈希冒充兼容"
    elif any(entry["kind"] == "path_hash_binding" for entry in refs):
        reason = "来源清单绑定路径和SHA256"
        condition = "阶段边界另建带版本的来源与阅读映射并核验；原冻结清单不回写"
    else:
        reason = "冻结设计、进展页或原位保护记录仍明确引用"
        condition = "纳入完整引用组或核验替代阅读与恢复入口后，再独立评估迁移；不改冻结原文"
    data = (root / name).read_bytes()
    protected.append(dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                          reason=reason, release_condition=condition, evidence=refs))
preflight = dict(schema_version=1, kind="RETROSPECTIVE_ARCHIVE_BATCH", state="PREFLIGHT_PASSED", source_commit=head,
    captured_at_utc=datetime.now(timezone.utc).isoformat(), previous_batch=prior_path,
    scope="Existing accepted/failed records 0000-0097, related early TASK1/TASK2 protocols, remaining early manuscript notes; Run0098+ excluded.",
    candidates=len(plan["candidates"]), candidate_sources=sources, moves=moves, protected=protected, maintained_files_before=maintained,
    preexisting_status=subprocess.check_output(["git", "status", "--porcelain=v1", "-uall"], cwd=root).decode("utf-8"),
    dependency_review=dict(frozen_reference_graph=True, fresh_markdown_roots=["reports", "paper", "agents", "docs", "designs", "campaigns", ".claude"],
        executable_basename_scan=True, closure_pruning=True,
        ignored_human_readers="Already archived originals use fixed-commit context; their bytes and references are not rewritten.",
        limits="Static analysis does not prove absence of arbitrary dynamic consumers; no executable, raw dataset or model input is moved."),
    json_structural_precheck=dict(files=21, errors=0, legacy_nonfinite_warnings=json_warnings),
    raw_data_arrays_or_weights_parsed=False, scientific_decisions_changed=False, experiments_run=False,
    allowed_changes=[entry["old_path"] for entry in moves] + [entry["new_path"] for entry in moves] + mutable + [
        "reports/history/private_rom/", "reports/history/early_e_series/catalog.md", "reports/history/early_e_series/decision_readers.md",
        "reports/history/paper/catalog.md", "reports/maintenance/early_gates/decision_readers.md",
        "results/repository_organization/20261108_early_history03/"])
(out / "preflight.json").write_text(json.dumps(preflight, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(dict(candidates=preflight["candidates"], moves=len(moves), protected=len(protected),
    json_precheck=preflight["json_structural_precheck"], groups=dict(Counter(entry["group"] for entry in moves)),
    git_byte_differences_only_in_line_endings=sum(not entry["git_blob_exact"] for entry in moves)), ensure_ascii=False))
