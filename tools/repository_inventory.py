"""Read-only stage-one inventory; only a new, explicit output directory is written."""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import io

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness_runtime.organization import Organization, read_public


PRIMARY = ("reports", "results", "experiments")
CONTEXT = ("campaigns", "harness", "designs", "docs", "src", "tests", "tools", "configs", "templates")
ROOT_FILES = ("AGENTS.md", "kiki.md", "README.md", "pyproject.toml", ".gitignore", ".gitattributes")
LOCAL = ("private_data", "artifacts", "local_backups")
MAX_TEXT_BYTES = 4 * 1024 * 1024
PATH_LITERAL = re.compile(r"(?<![\w/])((?:" + "|".join(PRIMARY + CONTEXT + LOCAL)
                          + r")/[^\s\"'<>`{}|,;()\[\]]+)")
LINK = re.compile(r"!?\[[^\]]*\]\(([^\s)]+)(?:\s+[\"'][^)]*)?\)")
METADATA_NAME = re.compile(r"release|manifest|receipt|adoption|archive|migration|protection|identity")


def linked(info):
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def relative(value):
    """Normalize repository paths lexically; never follow a referenced path."""
    value = value.replace("\\", "/")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ":" in value or ".." in path.parts or ".git" in path.parts:
        return None
    return path.as_posix().rstrip("/")


def safe_path(root, name):
    name = relative(name)
    if name is None:
        raise ValueError("path must stay inside the repository")
    path = root
    for part in PurePosixPath(name).parts:
        path = path / part
        try:
            if linked(path.lstat()):
                raise ValueError("refusing a symlink or reparse point: " + name)
        except FileNotFoundError:
            pass
    return path


def git(root, *args):
    result = subprocess.run(["git", "-c", "core.quotepath=false", *args], cwd=root,
                            capture_output=True, check=True)
    return result.stdout.decode("utf-8", errors="replace"), result.stderr.decode("utf-8", errors="replace")


def walk(root, folder):
    """Enumerate metadata without entering reparse points; report access failures."""
    path = root / folder
    try:
        info = path.lstat()
        if linked(info) or not stat.S_ISDIR(info.st_mode):
            yield folder, info, "link" if linked(info) else "file", None
            return
        with os.scandir(path) as entries:
            children = sorted(entries, key=lambda entry: entry.name)
    except OSError as exc:
        yield folder, None, "unreadable", type(exc).__name__
        return
    for entry in children:
        name = folder + "/" + entry.name
        try:
            info = entry.stat(follow_symlinks=False)
        except OSError as exc:
            yield name, None, "unreadable", type(exc).__name__
            continue
        if linked(info):
            yield name, info, "link", None
        elif stat.S_ISDIR(info.st_mode):
            if entry.name == ".git":
                yield name, info, "excluded_git_directory", None
            else:
                yield from walk(root, name)
        else:
            yield name, info, "file", None


def purpose(name):
    if "__pycache__" in PurePosixPath(name).parts:
        return "缓存候选"
    if name.startswith("experiments/"):
        return "实验入口或实验专用实现，待内容确认"
    if name.startswith("reports/"):
        return "报告、审查或冻结来源，待内容确认"
    if name.startswith("results/"):
        return "结果或执行记录，待内容确认"
    return "规范、登记、设计或调用者上下文"


def text_allowed(name, row):
    if row["kind"] != "file" or row["storage"] == "local_ignored" or row["bytes"] > MAX_TEXT_BYTES:
        return False
    suffix = PurePosixPath(name).suffix.lower()
    if name.startswith("results/"):
        # Read only explicit small metadata, never scientific arrays or model payloads.
        return suffix == ".json" and bool(METADATA_NAME.search(PurePosixPath(name).name))
    return suffix in {".md", ".py", ".ps1", ".json", ".toml"} or name in ROOT_FILES


def references(name, text, modules):
    rows = set()
    for number, line in enumerate(text.splitlines(), 1):
        for value in PATH_LITERAL.findall(line):
            target = relative(value.rstrip(".*。"))
            if target:
                rows.add((target, "literal_path_clue", number))
        if name.endswith(".md"):
            for value in LINK.findall(line):
                if ":" in value or value.startswith(("#", "/")):
                    continue
                value = value.split("#", 1)[0]
                if not value:
                    continue
                parts = list(PurePosixPath(name).parent.parts)
                for part in PurePosixPath(value.replace("\\", "/")).parts:
                    if part == "..":
                        if not parts:
                            break
                        parts.pop()
                    elif part != ".":
                        parts.append(part)
                else:
                    target = relative("/".join(parts))
                    if target:
                        rows.add((target, "markdown_link", number))
    if name.endswith(".py"):
        try:
            tree = ast.parse(text)
        except SyntaxError:
            rows.add((None, "python_parse_unresolved", 0))
        else:
            for node in ast.walk(tree):
                candidates = []
                if isinstance(node, ast.Import):
                    candidates = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    base = node.module or ""
                    if node.level:
                        package = name.removeprefix("src/").removesuffix(".py").split("/")[:-node.level]
                        base = ".".join(package + ([base] if base else []))
                    candidates = [base] + [base + "." + alias.name for alias in node.names]
                for module in candidates:
                    for target in modules.get(module, []):
                        rows.add((target, "python_import_clue", node.lineno))
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {
                        "glob", "rglob", "iterdir", "walk", "scandir"}:
                    rows.add((None, "dynamic_scan_requires_review", node.lineno))
    return [dict(source=name, target=target, kind=kind, line=line)
            for target, kind, line in sorted(rows, key=lambda row: (row[0] or "", row[1], row[2]))]


def hash_bindings(name, obj):
    """Recognize only explicit path-to-hash maps and path/hash metadata objects."""
    rows = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            target = relative(key) if isinstance(key, str) and "/" in key else None
            if target and isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value):
                rows.append(dict(source=name, target=target, kind="path_hash_binding", line=None))
            rows.extend(hash_bindings(name, value))
        path = obj.get("path")
        digest = obj.get("sha256")
        if isinstance(path, str) and isinstance(digest, str) and re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            target = relative(path)
            if target:
                rows.append(dict(source=name, target=target, kind="path_hash_binding", line=None))
    elif isinstance(obj, list):
        for value in obj:
            rows.extend(hash_bindings(name, value))
    return rows


def observe(root, name):
    try:
        info = safe_path(root, name).lstat()
        return dict(status="present", bytes=info.st_size, kind="directory" if stat.S_ISDIR(info.st_mode) else "file")
    except FileNotFoundError:
        return dict(status="absent")
    except PermissionError:
        return dict(status="inaccessible")
    except (OSError, ValueError) as exc:
        return dict(status="not_inspected", reason=type(exc).__name__)


def reconcile(root, access=None):
    rows = []
    organization = Organization(root, access)
    for entry in organization.migrations.values():
        rows.append(dict(entry, kind="documentation" if entry["source_record"] == organization.inputs.get("documentation_migration", "harness/documentation_migration_20261108.json") else "archive",
                         old_observation=observe(root, entry["old_path"]), new_observation=observe(root, entry["new_path"]),
                         current_verification="bytes_and_sha256_match"))
    name = "results/private_data_cleanup_20261107/migration_map.csv"
    path = safe_path(root, name)
    if path.is_file():
        with io.StringIO(read_public(root, name, access=access).decode("utf-8-sig")) as handle:
            reader = csv.reader(handle)
            headers = next(reader)
            if len(headers) != 6:
                raise ValueError("unexpected historical migration CSV schema")
            for values in reader:
                old, new, files, size = values[:4]
                if not old.startswith("private_data/") or not new.startswith(("artifacts/", "local_backups/")):
                    raise ValueError("unexpected historical migration roots")
                row = dict(kind="local_storage", source_record=name, old_path=old, new_path=new,
                           prior_files=int(files), prior_bytes=int(size),
                           old_observation=observe(root, old), new_observation=observe(root, new),
                           current_verification="not_verified", sha256_recheck=False)
                if row["new_observation"]["status"] == "present":
                    observed = list(walk(root, new))
                    complete = all(kind == "file" for _, _, kind, _ in observed)
                    row.update(observed_files=sum(kind == "file" for _, _, kind, _ in observed),
                               observed_bytes=sum(info.st_size for _, info, kind, _ in observed if kind == "file"),
                               skipped_or_unreadable=[dict(path=p, kind=k, error=e)
                                                      for p, _, k, e in observed if k != "file"])
                    row["current_verification"] = ("metadata_counts_and_bytes_match" if complete
                        and row["observed_files"] == row["prior_files"] and row["observed_bytes"] == row["prior_bytes"]
                        else "metadata_incomplete_or_different")
                rows.append(row)
    return rows


def inventory(root, output):
    root = root.absolute()
    out_name = relative(output)
    if not out_name or not out_name.startswith("results/repository_organization/"):
        raise ValueError("output must be a new directory under results/repository_organization/")
    out = safe_path(root, out_name)
    if out.exists():
        raise FileExistsError("refusing to overwrite an inventory")
    started = datetime.now(timezone.utc).isoformat()
    head, warnings = git(root, "rev-parse", "HEAD")
    branch, _ = git(root, "branch", "--show-current")
    status_before, _ = git(root, "status", "--porcelain=v1", "-uall")
    tracked, _ = git(root, "ls-files", "-z", "--cached")
    ignored, _ = git(root, "ls-files", "-z", "--others", "--ignored", "--exclude-standard", "--", *PRIMARY, *CONTEXT)
    tracked, ignored = set(tracked.split("\0")), set(ignored.split("\0"))
    records = {}
    for folder in PRIMARY + CONTEXT:
        if not (root / folder).exists():
            continue
        for name, info, kind, error in walk(root, folder):
            if name == out_name or name.startswith(out_name + "/"):
                continue
            records[name] = dict(original_path=name, current_path=name,
                scope="primary" if folder in PRIMARY else "context", kind=kind,
                bytes=info.st_size if info else None, mtime_ns=info.st_mtime_ns if info else None,
                suffix=PurePosixPath(name).suffix.lower(),
                storage="versioned" if name in tracked else "local_ignored" if name in ignored else "local_untracked",
                preliminary_purpose=purpose(name), purpose_basis="path_candidate_not_scientific_classification",
                owner=None, error=error, sha256=None)
    for name in ROOT_FILES:
        info = safe_path(root, name).stat()
        records[name] = dict(original_path=name, current_path=name, scope="context", kind="file",
            bytes=info.st_size, mtime_ns=info.st_mtime_ns, suffix=Path(name).suffix,
            storage="versioned" if name in tracked else "local_untracked", preliminary_purpose=purpose(name),
            purpose_basis="explicit_context", owner=None, error=None, sha256=None)
    modules = defaultdict(list)
    for name in records:
        if name.endswith(".py"):
            module = name.removeprefix("src/").removesuffix(".py").replace("/", ".").removesuffix(".__init__")
            modules[module].append(name)
            if name.startswith("experiments/"):
                modules[PurePosixPath(name).stem].append(name)
    relations, source_checks, parse_issues, access = [], {}, [], []
    for name, row in records.items():
        if not text_allowed(name, row):
            continue
        try:
            data = read_public(root, name, access=access)
            row["sha256"] = hashlib.sha256(data).hexdigest()
            source_checks[name] = row["sha256"]
            text = data.decode("utf-8-sig")
            relations.extend(references(name, text, modules))
            if name.endswith(".json"):
                relations.extend(hash_bindings(name, json.loads(text)))
        except (OSError, UnicodeError, ValueError, RecursionError) as exc:
            parse_issues.append(dict(path=name, error=type(exc).__name__))
    inbound = defaultdict(list)
    for edge in relations:
        if edge["target"]:
            inbound[edge["target"]].append(edge)
    migrations = reconcile(root, access)
    previous = defaultdict(list)
    for row in migrations:
        previous[row["new_path"]].append(row["old_path"])
    protected, pending = [], []
    for name, row in sorted(records.items()):
        parts = PurePosixPath(name).parts
        clues = [edge for i in range(1, len(parts) + 1) for edge in inbound.get("/".join(parts[:i]), [])
                 if edge["source"] != name]
        hard = [e for e in clues if e["kind"] in {"path_hash_binding", "python_import_clue"}
                or (e["source"].endswith((".py", ".ps1")) and e["kind"] == "literal_path_clue")]
        row.update(previous_paths=previous.get(name, []), dependency_clue_count=len(clues),
                   dependency_examples=clues[:5], move_authorized=False)
        if row["kind"] != "file":
            handling, reason = "not_inspected", "不可访问或链接目标未遍历；不能判为缺失或可移动"
        elif hard:
            handling, reason = "protected_in_place", "存在静态导入、源码路径或路径哈希绑定；现行使用范围仍需核实"
        elif clues or row["previous_paths"]:
            handling, reason = "ready_for_content_review", "已有联系或迁移来源，可继续核对内容；不代表路径依赖已排除"
        else:
            handling, reason = "needs_attribution_review", "归属及动态消费者尚未核清"
        row.update(handling=handling, reason=reason,
                   release_condition="核清现行与动态消费者、冻结身份及恢复入口后，另批确认路径调整")
        view = {k: row[k] for k in ("current_path", "handling", "reason", "release_condition", "dependency_clue_count")}
        if handling == "protected_in_place":
            protected.append(view)
        elif row["scope"] == "primary":
            pending.append(view)
    local_observations = []
    cleanup = root / "results/private_data_cleanup_20261107/summary.json"
    if cleanup.exists() or cleanup.is_symlink():
        prior = json.loads(read_public(root, cleanup.relative_to(root).as_posix(), access=access).decode("utf-8-sig"))
        for key in ("protected_roots_preserved", "unreadable_test_roots_preserved"):
            for directory in prior.get(key, []):
                name = "private_data/" + directory
                local_observations.append(dict(path=name, prior_category=key, observation=observe(root, name),
                    contents_checked=False, prior_record=cleanup.relative_to(root).as_posix(),
                    release_condition="复核既有保护清单、实际消费者及恢复能力；本批不解除保护"))
    changed = []
    for name, row in records.items():
        if row["kind"] != "file":
            continue
        try:
            info = safe_path(root, name).stat()
            if (info.st_size, info.st_mtime_ns) != (row["bytes"], row["mtime_ns"]):
                changed.append(dict(path=name, check="metadata_changed"))
            elif name in source_checks and hashlib.sha256(read_public(root, name, access=access)).hexdigest() != source_checks[name]:
                changed.append(dict(path=name, check="text_hash_changed"))
        except (OSError, ValueError) as exc:
            changed.append(dict(path=name, check=type(exc).__name__))
    status_after, _ = git(root, "status", "--porcelain=v1", "-uall")
    summary = dict(schema_version=1, kind="RETROSPECTIVE_FILE_INVENTORY", started_at_utc=started,
        completed_at_utc=datetime.now(timezone.utc).isoformat(), source_commit=head.strip(), branch=branch.strip(),
        coverage=list(PRIMARY), context=list(CONTEXT), excluded_output=out_name,
        entries=len(records), primary_entries=sum(r["scope"] == "primary" for r in records.values()),
        counts_by_root=dict(Counter(name.split("/")[0] for name in records)),
        counts_by_storage=dict(Counter(r["storage"] for r in records.values())),
        counts_by_handling=dict(Counter(r["handling"] for r in records.values())),
        dependency_clues=len(relations), dynamic_scan_clues=sum(e["kind"] == "dynamic_scan_requires_review" for e in relations),
        text_files_hashed=len(source_checks), unreadable_or_skipped=[dict(path=n, kind=r["kind"], error=r["error"])
            for n, r in records.items() if r["kind"] not in {"file"}], metadata_parse_issues=parse_issues,
        migrations_by_verification=dict(Counter(r["kind"] + ":" + r["current_verification"] for r in migrations)),
        original_file_changes_observed=changed, git_status_before=status_before, git_status_after=status_after,
        git_warnings=warnings, snapshot_atomic=False, read_scope_enforced=True,
        scientific_payloads_read=any(r["outcome"] == "read" and PurePosixPath(r["path"]).parts[0] in LOCAL for r in access),
        local_payload_hashes_recomputed=False, original_files_written=False,
        scientific_classification_performed=False, migration_or_deletion_authorized=False,
        live_process_state_verified=False,
        limitations=["静态依赖为线索，不证明现行调用或穷尽动态消费者", "未匹配引用不表示文件可移动",
                    "本地产物仅按旧迁移清单核对当前数量与大小；旧SHA256结果不冒充本次重验",
                    "私有保护根仅检查根路径；旧不可访问子树未重新遍历", "快照非原子；变化见逐项检查",
                    "mtime/大小相同的未读二进制内容没有做前后哈希保证"])
    out.mkdir(parents=True, exist_ok=False)
    def write_json(name, value):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    def write_rows(name, rows):
        with (out / name).open("x", encoding="utf-8", newline="\n") as handle:
            for value in rows:
                handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
    write_json("summary.json", summary)
    write_rows("inventory.jsonl", (records[name] for name in sorted(records)))
    write_rows("dependencies.jsonl", relations)
    write_rows("access.jsonl", access)
    write_rows("protected.jsonl", protected)
    write_rows("pending.jsonl", pending)
    write_rows("migration_reconciliation.jsonl", migrations)
    write_rows("local_roots.jsonl", local_observations)
    with (out / "inventory.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        fields = ["current_path", "scope", "kind", "bytes", "suffix", "storage", "preliminary_purpose", "handling", "dependency_clue_count"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records[name] for name in sorted(records))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = inventory(args.root, args.output)
    print(json.dumps({k: result[k] for k in ("entries", "primary_entries", "counts_by_root",
        "counts_by_handling", "migrations_by_verification", "original_file_changes_observed")}, ensure_ascii=False, indent=2))
