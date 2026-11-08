#!/usr/bin/env python3
"""Offline record checks and planning; never dispatch agents or execute science.

Examples (paths are supplied explicitly):
  python tools/research_harness.py assess --record harness/examples/run_single_matched_gain.json
  python tools/research_harness.py schedule --tasks harness/examples/tasks.json --resources '{"gpu:0":1}' --slots 3
  python tools/research_harness.py render-state --registry harness/examples/registry.snapshot.json
  python tools/research_harness.py check --root .

All commands are read-only unless a rendering command receives an explicit --output.
No live registry is required. Saved records are not live process verification or
execution authorization. The host must verify input hashes and current consent.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import stat
import sys
from urllib.parse import unquote, urlsplit

# Even local-module imports must not create bytecode on a read-only CLI call.
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness_runtime.scheduling import normalize_write_path, ready_tasks


REQUIRED_FILES = (
    "AGENTS.md", "README.md", "kiki.md",
    "agents/coordinator.md", "agents/experimenter.md", "agents/modeler.md",
    "agents/reviewer.md", "agents/runtime.md", "agents/scientist.md",
    "campaigns/README.md", "campaigns/retrospective_force_memory_20261030.md",
    "docs/architecture_design.md", "docs/current_design.md", "docs/research_policy.md",
    "docs/engineering_loop.md", "docs/harness_evolution.md",
    "templates/agent_task.md", "templates/architecture_brief.md", "templates/campaign.md",
    "templates/run_protocol.md", "templates/run_closure.md", "templates/mechanism_card.md",
    "harness/policy.json", "harness/examples/registry.snapshot.json",
    "harness/examples/run_functional_prediction_equivalence.json",
    "harness/examples/run_performance_equivalence.json", "harness/examples/run_reviewed_counterexample.json",
    "harness/examples/run_single_matched_gain.json", "harness/examples/run_single_negative.json",
    "harness/examples/run_two_seeds_not_null.json", "harness/examples/synthetic_evidence.md",
    "harness/examples/tasks.json",
    "harness_runtime/__init__.py", "harness_runtime/evidence.py",
    "harness_runtime/scheduling.py", "tools/research_harness.py",
    "tests/test_harness_cli.py", "tests/test_harness_evidence.py", "tests/test_harness_scheduling.py",
)
PRIVATE_PARTS = {"private_data", "private-data", "datasets", "checkpoints", "weights", "artifacts", "local_backups", ".git"}
TASK_STATES = {"RUNNING", "COMPLETED", "BLOCKED", "PAUSED_BY_USER", "INTERRUPTED_RUNTIME"}
LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^\s)]+)(?:\s+[\"'][^)]*)?\)")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def emit(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def schedule_snapshot(payload, resources, slots):
    """A paused envelope suppresses all ready suggestions, including read-only work."""
    if isinstance(payload, list):
        tasks, state = payload, None
    elif isinstance(payload, dict):
        tasks = payload.get("tasks")
        state = payload.get("task_state", payload.get("status"))
    else:
        tasks, state = payload, None
    result = ready_tasks(tasks, resources, slots)
    result["snapshot_state"] = state
    result["live_state_verified"] = False
    result["paused"] = state == "PAUSED_BY_USER"
    if state is not None and (not isinstance(state, str) or state not in TASK_STATES):
        result["issues"].append({"severity": "error", "code": "invalid_task_state", "message": "unknown envelope task_state"})
        result["valid"] = False
        result["ready"] = []
    if result["paused"]:
        # Recompute occupancy with pending work paused so no suggested allocations
        # remain in the resource/slot report after the ready group is suppressed.
        if isinstance(tasks, list):
            paused_tasks = [dict(t, status="paused") if isinstance(t, dict) and t.get("status") == "pending" else t for t in tasks]
            occupancy = ready_tasks(paused_tasks, resources, slots)
            for field in ("resources", "slots"):
                if field in occupancy:
                    result[field] = occupancy[field]
            for task in tasks:
                if isinstance(task, dict) and task.get("status") == "pending" and isinstance(task.get("task_id"), str):
                    result["blocked"][task["task_id"]] = ["snapshot_paused_by_user"]
        result["ready"] = []
        result["notice"] = "Saved PAUSED_BY_USER snapshot: no task is suggested for launch. This utility cannot resume work."
    else:
        result["notice"] = "Planning candidates only, including for a bare task list. No live consent, input hash, resource lease, or process state was verified."
    return result


def render_state(registry):
    if not isinstance(registry, dict):
        raise ValueError("registry must be an object")
    state = registry.get("task_state", registry.get("status"))
    if not isinstance(state, str) or state not in TASK_STATES:
        raise ValueError("registry task_state is required and must use a known task state")
    snapshot = registry.get("snapshot", {})
    current = registry.get("current_run", {})
    jobs = registry.get("active_jobs")
    pointers = registry.get("pointers", {})
    if not isinstance(snapshot, dict) or not isinstance(current, dict) or not isinstance(pointers, dict):
        raise ValueError("snapshot/current_run/pointers must be objects")
    if jobs is not None and not isinstance(jobs, list):
        raise ValueError("active_jobs must be a list, null, or absent")
    job_summary = ("The snapshot's active job count is unknown." if jobs is None
                   else f"The snapshot records {len(jobs)} active job(s).")

    def text(value):
        if value is None:
            return "unknown"
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True)
        return str(value).replace("\r", " ").replace("\n", " ").replace("|", "\\|")

    lines = [
        "# Research harness snapshot", "",
        "> Saved evidence snapshot only. This command does not verify live processes, resume work, or grant execution permission.", "",
        "- Task state: **" + text(state) + "**",
        "- Snapshot time (UTC): " + text(snapshot.get("as_of_utc", registry.get("snapshot_at_utc"))),
        "- Source commit: " + text(snapshot.get("source_sha", registry.get("source_sha"))),
        "- Live verification performed by this command: **no**", "",
        "## Current Run", "",
        "- Run: " + text(current.get("run_id")),
        "- Recorded status: " + text(current.get("status")), "",
        "## Active jobs in the supplied snapshot", "",
        job_summary + " Live process state was not inspected.", "",
        "## Pointers", "", "| Pointer | Recorded value |", "|---|---|",
    ]
    for key, value in pointers.items():
        lines.append("| " + text(key) + " | " + text(value) + " |")
    if not pointers:
        lines.append("| none supplied | unknown |")
    lines += ["", "## Next action", "", text(registry.get("next_action"))]
    if state == "PAUSED_BY_USER":
        lines += ["", "Scientific execution remains paused. This snapshot is not permission to start another Run."]
    notes = registry.get("notes", [])
    if isinstance(notes, list) and notes:
        lines += ["", "## Notes", ""] + ["- " + text(note) for note in notes]
    return "\n".join(lines) + "\n"


def _private(path):
    return any(part.lower() in PRIVATE_PARTS for part in path.parts)


def _link_or_reparse(path):
    """Reject Windows junctions/reparse points as well as ordinary symlinks."""
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    return (stat.S_ISLNK(info.st_mode)
            or bool(getattr(info, "st_file_attributes", 0)
                    & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)))


def _inside(path, root):
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _safe_public_path(path, root):
    relative = path.relative_to(root)
    if _private(relative):
        return False
    # Inspect parents before resolving: a junction may be a cycle, and a
    # non-symlink leaf can still be reached through a reparse-point parent.
    cursor = root
    for part in relative.parts:
        cursor /= part
        if _link_or_reparse(cursor):
            return False
    return _inside(path, root) and not _private(path.resolve().relative_to(root.resolve()))


def _controlled_files(directory):
    """Walk public configuration without following links or Windows junctions."""
    if _link_or_reparse(directory) or not directory.exists():
        return
    for path in sorted(directory.iterdir()):
        if _private(path) or _link_or_reparse(path):
            continue
        if path.is_dir():
            yield from _controlled_files(path)
        elif path.suffix.lower() in {".json", ".md", ".py"}:
            yield path


def _json_references(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "references" and isinstance(item, list):
                yield from (x for x in item if isinstance(x, str))
            elif key.endswith("_reference") and isinstance(item, str):
                yield item
            else:
                yield from _json_references(item)
    elif isinstance(value, list):
        for item in value:
            yield from _json_references(item)


def run_navigation(root_path, run_id=None):
    """Read registered identities and explicit links; never open scientific payloads."""
    root = Path(root_path).resolve()
    if run_id is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id):
        raise ValueError("run_id must be a single identifier")
    index_path = root / "harness/navigation.json"
    index = {}
    if index_path.exists() or _link_or_reparse(index_path):
        if not _safe_public_path(index_path, root):
            raise ValueError("unsafe navigation index")
        index = load_json(index_path)
        if not isinstance(index, dict) or index.get("kind") != "RETROSPECTIVE_NAVIGATION" or not isinstance(index.get("runs"), dict):
            raise ValueError("navigation requires RETROSPECTIVE_NAVIGATION and runs")
    run_dir = root / "harness/runs"
    if not _safe_public_path(run_dir, root) or not run_dir.is_dir():
        raise ValueError("public Run registry directory is absent or unsafe")
    paths = [run_dir / (run_id + ".json")] if run_id else sorted(run_dir.glob("*.json"))
    migrations = {}
    migration_path = root / "harness/documentation_migration_20261108.json"
    if migration_path.exists() or _link_or_reparse(migration_path):
        if not _safe_public_path(migration_path, root):
            raise ValueError("unsafe documentation migration")
        migration = load_json(migration_path)
        for entry in migration.get("entries", []):
            if entry["old_path"] in migrations:
                raise ValueError("duplicate documentation migration")
            migrations[entry["old_path"]] = entry
    result = dict(valid=True, runs=[], issues=[], read_only=True, private_payloads_read=False,
                  live_state_verified=False, scientific_evidence_verified=False)

    def reference(name, field, follow_migration=True):
        row = dict(reference=name, field=field, status="unknown")
        if (not isinstance(name, str) or not name or ":" in name or "\\" in name or re.search(r"[%?#\x00-\x1f]", name)
                or any(part in {"", ".", ".."} for part in name.split("/"))):
            row["status"] = "unsafe"
        elif _private(Path(name)):
            row["status"] = "local_not_checked"
        elif not _safe_public_path(root / name, root):
            row["status"] = "unsafe"
        else:
            row["status"] = "present" if (root / name).exists() else "missing"
            if row["status"] == "missing" and follow_migration and name in migrations:
                entry = migrations[name]
                if not isinstance(entry.get("new_path"), str) or not entry["new_path"].endswith(".md") or entry["new_path"].split("/")[0] not in {"reports", "designs"}:
                    raise ValueError("migration target must be public Markdown documentation")
                target = reference(entry["new_path"], "documentation_migration", False)
                if target["status"] == "present":
                    payload = (root / entry["new_path"]).read_bytes()
                    if len(payload) != entry["bytes"] or hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                        raise ValueError("migrated documentation bytes differ: " + name)
                    row.update(status="migrated_sha256_verified", resolved_reference=entry["new_path"])
        if row["status"] in {"missing", "unsafe"}:
            result["issues"].append(dict(code=row["status"] + "_reference", **row))
            result["valid"] = False
        return row

    def references(value, field=""):
        if isinstance(value, dict):
            for key, item in value.items():
                label = field + "." + key if field else key
                if key == "reference" or key.endswith("_reference"):
                    if isinstance(item, str):
                        yield reference(item, label)
                elif key == "references" or key.endswith("_references"):
                    if isinstance(item, list):
                        for i, ref in enumerate(item):
                            if isinstance(ref, str):
                                yield reference(ref, f"{label}[{i}]")
                else:
                    yield from references(item, label)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                yield from references(item, f"{field}[{i}]")

    for path in paths:
        if not _safe_public_path(path, root) or not path.is_file():
            raise ValueError("Run record is absent or unsafe: " + path.name)
        record = load_json(path)
        if not isinstance(record, dict) or record.get("run_id") != path.stem:
            raise ValueError("Run identity does not match its registry filename")
        nav = index.get("runs", {}).get(path.stem, {})
        if not isinstance(nav, dict):
            raise ValueError("Run navigation must be an object")
        if not isinstance(nav.get("events", []), list) or any(not isinstance(e, dict) for e in nav.get("events", [])):
            raise ValueError("navigation events must be a list of objects")
        for section, required in (("materials", {"reference", "role"}), ("events", {"reference"}),
                                  ("corrections", {"original_reference", "corrected_by_reference", "adopted_reference", "scope"})):
            rows = nav.get(section, [])
            if not isinstance(rows, list) or any(not isinstance(row, dict) or not required <= row.keys()
                    or any(not isinstance(row[key], str) for key in required) for row in rows):
                raise ValueError("invalid navigation " + section)
        execution = record.get("execution", {})
        decisions = record.get("root_decisions", {})
        work = record.get("scientific_work", {})
        proposal = record.get("proposal", {})
        if not all(isinstance(v, dict) for v in (execution, decisions, work, proposal)):
            raise ValueError("execution/decisions/work/proposal must be objects")
        states = dict(execution_status=execution.get("status"), budget_status=work.get("budget_status"),
            code_decision=decisions.get("code_decision", decisions.get("code")),
            model_decision=decisions.get("model_decision", decisions.get("model")),
            science_decision=decisions.get("science_decision", decisions.get("science")),
            mechanism_claim=decisions.get("mechanism_claim_status"),
            scientific_claim_proposed=proposal.get("scientific_claim"))
        conflicts = []
        for a, b in ((work.get("budget_status"), decisions.get("scientific_budget_status")),
                     (decisions.get("code"), decisions.get("code_decision")),
                     (decisions.get("model"), decisions.get("model_decision")),
                     (decisions.get("science"), decisions.get("science_decision"))):
            if a is not None and b is not None and a != b:
                conflicts.append(dict(values=[a, b], resolution="not_selected"))
        if conflicts:
            result["valid"] = False
            result["issues"].append(dict(code="record_state_conflict", run_id=path.stem, conflicts=conflicts))
        materials = list(references(record)) + list(references(nav, "navigation"))
        events = []
        # Only explicitly labelled execution metadata is opened, never arbitrary
        # result JSON, arrays, weights, or paths obtained from those receipts.
        receipt_names = {"execution.json", "execution_receipt.json", "execution_host_receipt.json",
                         "resume_execution_receipt.json", "resume_execution_host_receipt.json",
                         "archive_verified.json", "root_adoption.json"}
        for event in nav.get("events", []):
            ref = event.get("reference")
            checked = reference(ref, "navigation.events")
            row = dict(event, status=checked["status"])
            if checked["status"] == "present":
                target = root / ref
                if not ref.startswith("results/") or target.name not in receipt_names or target.stat().st_size > 1048576:
                    raise ValueError("event must name a small public execution/decision/archive receipt")
                body = load_json(target)
                if not isinstance(body, dict):
                    raise ValueError("event receipt must be an object")
                if body.get("run_id") not in {None, path.stem}:
                    result["valid"] = False
                    result["issues"].append(dict(code="event_identity_conflict", reference=ref))
                decision = body.get("root_decisions", {})
                row["recorded_utc"] = body.get("recorded_utc", decision.get("recorded_utc") if isinstance(decision, dict) else None)
                row["times"] = {k: body[k] for k in ("started_utc", "completed_utc", "finished_utc", "host_start_utc", "host_end_utc") if k in body}
                row["saved_status"] = body.get("status")
                row["exit_code"] = body.get("exit_code")
            events.append(row)
        result["runs"].append(dict(run_id=path.stem, title=nav.get("title", path.stem),
            record_reference=path.relative_to(root).as_posix(), recorded_utc=record.get("recorded_utc"),
            purpose=record.get("purpose"), scope=record.get("scope"), states=states,
            proposal=proposal, navigation=nav, materials=materials, events=events, conflicts=conflicts,
            unknown_fields=[k for k, v in states.items() if v is None]))
    return result


def render_run_navigation(value):
    """Render a deterministic reading view; state remains in the cited records."""
    def text(value):
        return ("unknown" if value is None else str(value)).replace("\n", " ").replace("|", "\\|")

    lines = ["# 已登记研究材料导航", "", "记录状态来自所列登记，不代表实时进程核验；未读取本地模型、数据或预测数组。", ""]
    if len(value["runs"]) > 1:
        lines += ["单个 Run 的完整材料与事件关系可用 `show-run --run <编号>` 查看；本表只显示原登记值。", "",
                  "| 中文问题 | 追溯编号／登记 | 执行 | 预算 | 模型决定 | 科学决定 |", "|---|---|---|---|---|---|"]
        for run in value["runs"]:
            state = run["states"]
            lines.append(f"| {text(run['title'])} | [{run['run_id']}]({run['record_reference']}) | {text(state['execution_status'])} | {text(state['budget_status'])} | {text(state['model_decision'])} | {text(state['science_decision'])} |")
        lines += ["", "旧字段未记录的值显示 unknown，不用提议决定补成 root 采用决定。"]
        if value["issues"]:
            lines += ["", "## 待核查", ""] + ["- " + text(row) for row in value["issues"]]
        return "\n".join(lines) + "\n"
    for run in value["runs"]:
        nav = run["navigation"]
        lines += ["## " + text(run["title"]), "", f"追溯编号：`{run['run_id']}`；[登记]({run['record_reference']})；记录时间：{text(run['recorded_utc'])}。", "",
                  "| 判断 | 登记值 |", "|---|---|"]
        labels = {"execution_status": "执行", "budget_status": "工作量／预算", "code_decision": "代码决定",
                  "model_decision": "模型决定", "science_decision": "科学决定", "mechanism_claim": "机制主张",
                  "scientific_claim_proposed": "提议主张（不作为采用决定）"}
        lines += [f"| {labels[k]} | {text(v)} |" for k, v in run["states"].items()]
        if nav.get("campaign_reference"):
            lines += ["", f"[主题综合]({nav['campaign_reference']})；父 Run：{text(nav.get('parent_run'))}。"]
        if nav.get("materials"):
            lines += ["", "### 主要阅读材料", ""]
            for row in nav["materials"]:
                protection = row.get("path_protection", {})
                status = "原位保留" if protection.get("status") == "preserve_current_path" else text(protection.get("status"))
                lines.append(f"- {text(row.get('role'))}：[{text(row['reference'])}]({row['reference']})；路径处理：{status}，{text(protection.get('reason'))}。")
            if nav.get("protection_release_condition"):
                lines += ["", "解除条件：" + text(nav["protection_release_condition"])]
        for correction in nav.get("corrections", []):
            lines += ["", "### 解释更正", "", text(correction.get("scope")), "",
                      f"[原解释]({correction['original_reference']}) → [独立更正]({correction['corrected_by_reference']}) → [采用更正的结论]({correction['adopted_reference']})。"]
        lines += ["", "### 执行与归档事件", ""]
        for event in run["events"]:
            time_labels = {"started_utc": "启动", "host_start_utc": "宿主启动", "completed_utc": "完成", "finished_utc": "结束", "host_end_utc": "宿主结束"}
            times = "；".join(time_labels[k] + "：" + text(v) for k, v in event.get("times", {}).items()) or "未记录执行时刻"
            saved = "；保存状态：" + text(event["saved_status"]) if event.get("saved_status") is not None else ""
            lines.append(f"- {text(event.get('label'))}：[回执]({event['reference']})；记录时间：{text(event.get('recorded_utc'))}；{times}{saved}；关系：{text(event.get('relation'))}。")
        if not run["events"]:
            lines.append("未补录明确的执行事件关系；不按文件名推断。")
        lines += ["", "### 材料与字段来源", "", "仅有平铺引用的材料保留为其他已登记材料，不推断其科学用途。路径检查不核验科学内容。", "",
                  "本页只建立导航，原件保持原位；协议、审查与源码的身份绑定沿执行释放清单核查，迁移前须解除依赖。", "",
                  "| 字段来源 | 材料 | 路径检查 |", "|---|---|---|"]
        grouped = {}
        for row in run["materials"]:
            grouped.setdefault(row["reference"], []).append(row)
        for name, rows in grouped.items():
            fields = "; ".join(dict.fromkeys(row["field"] for row in rows))
            row = rows[0]
            lines.append(f"| {text(fields)} | [{text(name)}]({row.get('resolved_reference', name)}) | {row['status']} |")
        if run["conflicts"]:
            lines += ["", "存在记录冲突，未自动选择其中一个值。"]
        lines.append("")
    if value["issues"]:
        lines += ["## 待核查", ""] + ["- " + text(row) for row in value["issues"]]
    return "\n".join(lines).rstrip() + "\n"


def check_root(root_path):
    """Check the overlay contract and explicit public metadata, not scientific data.

    Accepts a package containing overlay/, an overlay directory, or an installed
    repository. A package overlay may refer to pre-existing repository reports;
    absent non-bundled references are reported as warnings. Within the harness
    and template directories, missing references are errors. No active registry
    is required and no private/data/checkpoint payload is opened.
    """
    requested = Path(root_path).resolve()
    if _private(requested) or _link_or_reparse(requested / "overlay"):
        return {
            "valid": False, "issues": [{"severity": "error", "code": "unsafe_check_root", "message": "check root must be public metadata, without a symlinked overlay"}],
            "checked_files": [], "checked_relative_references": [], "root": str(requested),
            "read_only": True, "active_registry_required": False, "private_payloads_read": False,
        }
    package_mode = (requested / "overlay").is_dir()
    root = requested / "overlay" if package_mode else requested
    issues, checked, refs = [], [], []

    def issue(code, message, path=None, severity="error"):
        row = {"severity": severity, "code": code, "message": message}
        if path is not None:
            row["path"] = str(path)
        issues.append(row)

    # This fixed map preserves documentation provenance, never scientific authority.
    migrations = {}
    migration = root / "harness/documentation_migration_20261108.json"
    if migration.exists() or _link_or_reparse(migration):
        try:
            if not _safe_public_path(migration, root):
                raise ValueError("migration map must be a safe public local file")
            body = load_json(migration)
            if (not isinstance(body, dict) or type(body.get("schema_version")) is not int
                    or body["schema_version"] != 1
                    or not isinstance(body.get("source_commit"), str) or not body["source_commit"]
                    or not isinstance(body.get("entries"), list)):
                raise ValueError("migration requires schema_version=1, source_commit and entries")
            seen = set()
            for entry in body["entries"]:
                if not isinstance(entry, dict):
                    raise ValueError("migration entry must be an object")
                old, new = entry.get("old_path"), entry.get("new_path")
                for name, prefixes in ((old, {"docs", "reports"}), (new, {"reports", "designs"})):
                    if (not isinstance(name, str) or "\\" in name or ":" in name
                            or not name.endswith(".md")
                            or name.split("/")[0] not in prefixes
                            or len(name.split("/")) < 2
                            or any(part in {"", ".", ".."} for part in name.split("/"))
                            or not _safe_public_path(root / name, root)):
                        raise ValueError("migration paths must be canonical public docs/reports sources and reports/designs targets")
                if old in seen:
                    raise ValueError("duplicate migration old_path: " + old)
                seen.add(old)
                digest, size = entry.get("sha256"), entry.get("bytes")
                if (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest)
                        or type(size) is not int or size < 0):
                    raise ValueError("migration requires sha256 and nonnegative integer bytes")
                target = root / new
                if not target.is_file():
                    raise ValueError("migration target is absent: " + new)
                payload = target.read_bytes()
                if len(payload) != size or hashlib.sha256(payload).hexdigest() != digest.lower():
                    raise ValueError("migration target bytes differ: " + new)
                migrations[old] = target
        except (OSError, ValueError, RuntimeError) as exc:
            migrations.clear()
            issue("invalid_documentation_migration", str(exc), migration)
        approved_docs = {"architecture_design.md", "engineering_loop.md", "research_policy.md",
                         "harness_evolution.md", "current_design.md", "current_state.md"}
        docs = root / "docs"
        if not _safe_public_path(docs, root) or not docs.is_dir():
            issue("invalid_documentation_scope", "docs must contain exactly six approved public files", docs)
        else:
            actual_docs = {path.name for path in docs.iterdir()}
            if actual_docs != approved_docs or any(
                    not _safe_public_path(docs / name, root) or not (docs / name).is_file()
                    for name in approved_docs):
                issue("invalid_documentation_scope", "docs must contain exactly six approved public files", docs)

    def reference(ref, source, json_ref=False):
        if not ref or ref.startswith("#") or "<" in ref or ">" in ref:
            return
        parsed = urlsplit(ref)
        if parsed.scheme in {"http", "https", "mailto", "app"}:
            return
        if parsed.scheme or ref.startswith("/") or "\\" in ref:
            issue("unsafe_reference", "reference must be a relative path or an ordinary web URL", source)
            return
        path = unquote(parsed.path)
        # JSON evidence references use repository-relative paths. Markdown links
        # use the containing document's directory, as GitHub does.
        target = (root if json_ref else source.parent) / path
        try:
            inside = _inside(target, root)
        except (OSError, RuntimeError) as exc:
            issue("reference_resolution_error", "cannot resolve reference " + ref + ": " + str(exc), source)
            return
        if not inside:
            issue("reference_outside_root", "reference escapes the repository: " + ref, source)
            return
        if _private(target.relative_to(root)):
            issue("private_reference_not_read", "opaque private reference was not opened or checked: " + ref, source, "warning")
            return
        refs.append({"source": source.relative_to(root).as_posix(), "reference": ref})
        if not target.exists():
            # Active paths always win; frozen records and manifests remain untouched.
            old = target.resolve().relative_to(root.resolve()).as_posix()
            if old in migrations:
                return
            bundled = target.relative_to(root).parts[:1] in [("harness",), ("harness_runtime",), ("tools",), ("templates",)]
            severity = "warning" if package_mode and not bundled else "error"
            issue("missing_reference", "relative reference is absent: " + ref, source, severity)

    files = set()
    if migration.exists():
        files.add(root / "docs/current_state.md")
    for name in REQUIRED_FILES:
        path = root / name
        if not _safe_public_path(path, root) or not path.is_file():
            issue("missing_required_file", "required public file is missing or not a safe local file", name)
        else:
            files.add(path)
    files.update(_controlled_files(root / "harness") or [])

    # Manifest paths are overlay/repository-relative, even when --root names the
    # outer package. Only explicitly listed public artifacts are hashed. base_*
    # describes the installation precondition, not the bytes of this new overlay.
    # No scientific input hash or evidence reference is followed for hashing.
    manifest = requested / "manifest.json"
    if not _link_or_reparse(manifest) and manifest.is_file():
        try:
            body = load_json(manifest)
            if not isinstance(body, dict):
                raise ValueError("manifest must be an object")
            entries = body.get("files", [])
            if not isinstance(entries, list):
                issue("invalid_manifest", "manifest files must be a list", manifest)
            else:
                for entry in entries:
                    name = entry.get("path") if isinstance(entry, dict) else entry
                    if not isinstance(name, str):
                        issue("invalid_manifest_path", "manifest file path must be text", manifest)
                        continue
                    try:
                        normalized_name = normalize_write_path(name)
                    except ValueError as exc:
                        issue("unsafe_manifest_path", str(exc), name)
                        continue
                    target = root / normalized_name
                    if not _safe_public_path(target, root):
                        issue("unsafe_manifest_path", "manifest paths must stay inside the overlay and outside private payloads", name)
                    elif not target.is_file() or _link_or_reparse(target):
                        issue("missing_manifest_file", "explicit manifest file is absent or a symlink", name)
                    else:
                        if isinstance(entry, dict) and ("new_sha256" in entry or "bytes" in entry):
                            payload = target.read_bytes()
                            expected_hash = entry.get("new_sha256")
                            if "new_sha256" in entry:
                                if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash):
                                    issue("invalid_manifest_hash", "new_sha256 must be a 64-character hexadecimal digest", name)
                                elif hashlib.sha256(payload).hexdigest() != expected_hash.lower():
                                    issue("manifest_hash_mismatch", "overlay bytes do not match new_sha256", name)
                            if "bytes" in entry:
                                expected_size = entry["bytes"]
                                if type(expected_size) is not int or expected_size < 0:
                                    issue("invalid_manifest_size", "bytes must be a nonnegative integer", name)
                                elif len(payload) != expected_size:
                                    issue("manifest_size_mismatch", "overlay byte count differs from manifest", name)
                        if target.suffix.lower() in {".json", ".md", ".py"}:
                            files.add(target)
        except (OSError, ValueError, RuntimeError) as exc:
            issue("invalid_manifest", str(exc), manifest)

    for path in sorted(files):
        if not _safe_public_path(path, root):
            issue("unsafe_metadata_path", "public metadata path is outside the allowed scope", path)
            continue
        checked.append(path.relative_to(root).as_posix())
        try:
            content = path.read_text(encoding="utf-8")
            suffix = path.suffix.lower()
            if suffix == ".json":
                data = json.loads(content)
                if path == root / "harness/policy.json":
                    if not isinstance(data, dict) or data.get("schema_version") != 2 or not isinstance(data.get("permissions"), dict):
                        issue("invalid_policy", "policy requires schema_version=2 and an object permissions; additional fields are allowed", path)
                if isinstance(data, dict) and "task_state" in data and "current_run" in data:
                    try:
                        render_state(data)
                    except ValueError as exc:
                        issue("invalid_snapshot", str(exc), path)
                if isinstance(data, dict) and "tasks" in data:
                    tasks = data["tasks"]
                    capacities = {}
                    if isinstance(tasks, list):
                        for task in tasks:
                            if isinstance(task, dict) and isinstance(task.get("resources"), dict):
                                for name, quantity in task["resources"].items():
                                    if isinstance(name, str) and isinstance(quantity, int) and not isinstance(quantity, bool) and quantity >= 0:
                                        capacities[name] = capacities.get(name, 0) + quantity
                    validation = schedule_snapshot(data, capacities, len(tasks) if isinstance(tasks, list) else 0)
                    for row in validation["issues"]:
                        issue(row["code"], row["message"], path, row.get("severity", "error"))
                for ref in _json_references(data):
                    reference(ref, path, json_ref=True)
            elif suffix == ".md":
                for ref in LINK_PATTERN.findall(content):
                    reference(ref, path)
            elif suffix == ".py":
                ast.parse(content, filename=str(path))
        except (OSError, UnicodeError, ValueError, SyntaxError, RuntimeError) as exc:
            issue("invalid_public_metadata", str(exc), path)
    return {
        "valid": not any(row["severity"] == "error" for row in issues), "issues": issues,
        "checked_files": checked, "checked_relative_references": refs,
        "root": str(root), "package_mode": package_mode, "read_only": True,
        "manifest_paths_relative_to": "overlay/repository root", "base_hashes_checked": False,
        "active_registry_required": False, "private_payloads_read": False,
        "notice": "Static public-metadata check only; no scientific evidence, model, live state, authority, or input hash was certified.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    assess = commands.add_parser("assess", help="assess a supplied record; exit 0 means schema valid, not scientific acceptance")
    assess.add_argument("--record", required=True)
    schedule = commands.add_parser("schedule", help="report feasible planning candidates; never dispatch")
    schedule.add_argument("--tasks", required=True)
    schedule.add_argument("--resources", required=True, help="JSON resource capacities supplied by the host")
    schedule.add_argument("--slots", required=True, type=int, help="available host agent slots, including running tasks")
    render = commands.add_parser("render-state", help="render a saved snapshot, stdout by default")
    render.add_argument("--registry", required=True)
    render.add_argument("--output", help="explicit destination for the rendered snapshot")
    check = commands.add_parser("check", help="check public package metadata without an installed active registry")
    check.add_argument("--root", required=True)
    check.add_argument("--run", help="check one registered Run and its explicit public links")
    show = commands.add_parser("show-run", help="show registered material links without executing science")
    show.add_argument("--root", required=True)
    selected = show.add_mutually_exclusive_group(required=True)
    selected.add_argument("--run")
    selected.add_argument("--all", action="store_true")
    show.add_argument("--format", choices=("json", "markdown"), default="markdown")
    show.add_argument("--output", help="explicit new destination; existing files are never overwritten")
    args = parser.parse_args(argv)
    try:
        if args.command == "assess":
            from harness_runtime.evidence import assess_record
            result = assess_record(load_json(args.record))
        elif args.command == "schedule":
            result = schedule_snapshot(load_json(args.tasks), json.loads(args.resources), args.slots)
        elif args.command == "render-state":
            rendered = render_state(load_json(args.registry))
            if args.output:
                target = Path(args.output)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(rendered, encoding="utf-8")
                emit({"written": str(target), "live_state_verified": False, "dispatch_performed": False})
            else:
                print(rendered, end="")
            return 0
        elif args.command == "show-run":
            result = run_navigation(args.root, args.run)
            rendered = render_run_navigation(result) if args.format == "markdown" else json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            if args.output:
                target = Path(args.output)
                # Markdown links are rooted here, regardless of the output directory.
                if args.format == "markdown":
                    base = target.resolve().parent
                    import os
                    def relocate_link(match):
                        link = Path(os.path.relpath(Path(args.root).resolve() / match.group(1), base)).as_posix()
                        start, end = match.start(1) - match.start(), match.end(1) - match.start()
                        return match.group(0)[:start] + link + match.group(0)[end:]
                    rendered = LINK_PATTERN.sub(relocate_link, rendered)
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("x", encoding="utf-8", newline="\n") as stream:
                    stream.write(rendered)
            else:
                print(rendered, end="")
            return 0 if result["valid"] else 1
        elif args.command == "check" and args.run:
            result = run_navigation(args.root, args.run)
        else:
            result = check_root(args.root)
        emit(result)
        return 0 if result.get("valid") is True else 1
    except (OSError, UnicodeError, ValueError, TypeError, RuntimeError) as exc:
        emit({"valid": False, "issues": [{"severity": "error", "code": "input_error", "message": str(exc)}], "dispatch_performed": False})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
