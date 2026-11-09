"""Public organization metadata and verified archive aliases; never read local payloads."""
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess

PRIVATE_PARTS = {"private_data", "private-data", "artifacts", "local_backups", "datasets", "checkpoints", "weights", ".git"}
PUBLIC_ROOTS = {"reports", "results", "experiments", "harness", "campaigns", "designs", "docs", "src", "tests", "tools", "configs", "templates", "agents", "harness_runtime"}
METADATA_NAME = re.compile(r"release|manifest|receipt|adoption|archive|migration|protection|identity|summary|execution")
CURRENT = "results/repository_organization/current.json"
DOCUMENTS = "harness/documentation_migration_20261108.json"
MAX_BYTES = 4 * 1024 * 1024
ROOT_FILES = {"AGENTS.md", "kiki.md", "README.md", "pyproject.toml", ".gitignore", ".gitattributes"}


def canonical(name):
    if (not isinstance(name, str) or not name or re.search(r"[\\:%?#\x00-\x1f]", name)
            or any(p in {"", ".", ".."} or p.endswith((".", " ")) for p in name.split("/"))):
        raise ValueError("path must be a canonical repository-relative path")
    return name


def public_path(root, name):
    canonical(name)
    parts = PurePosixPath(name).parts
    if any(p.lower() in PRIVATE_PARTS for p in parts):
        raise ValueError("private payload path is not readable: " + name)
    root = Path(root).absolute()
    if any(p.lower() in PRIVATE_PARTS for p in root.parts):
        raise ValueError("private repository root is not readable")
    cursor = root
    for part in ("", *parts):
        cursor = cursor / part
        try:
            info = cursor.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("refusing symlink or reparse point: " + name)
    return cursor


def read_public(root, name, *, kind="metadata", access=None):
    """Every content read checks scope, type, size and all path components."""
    row = dict(path=name, kind=kind, outcome="refused")
    try:
        path = public_path(root, name)
        parts = PurePosixPath(name).parts
        if ((len(parts) > 1 and parts[0] not in PUBLIC_ROOTS)
                or len(parts) == 1 and name not in ROOT_FILES):
            raise ValueError("unapproved public scope: " + name)
        if kind not in {"metadata", "documentation", "archive"}:
            raise ValueError("unknown public material type")
        suffix = path.suffix.lower()
        if kind == "documentation":
            allowed = parts[0] in {"docs", "reports", "designs"} and suffix == ".md"
        elif kind == "archive":
            allowed = (parts[0] in {"docs", "reports", "designs", "experiments", "results"}
                       and suffix in {".md", ".py", ".json", ".txt", ".html", ".diff"})
            if parts[0] == "results":
                allowed = suffix == ".json" and len(parts) > 2 and (parts[1].endswith("_history") or parts[1] == "early_e_series_evidence")
        else:
            allowed = suffix in {".md", ".py", ".ps1", ".json", ".csv", ".toml"} or len(parts) == 1
            if parts[0] == "results":
                allowed = ((name.startswith("results/repository_organization/") and suffix in {".json", ".csv", ".md"})
                           or suffix == ".json" and bool(METADATA_NAME.search(path.name))
                           or name == "results/private_data_cleanup_20261107/migration_map.csv")
        if not allowed:
            raise ValueError("unapproved file type or material scope: " + name)
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_BYTES:
            raise ValueError("not a small regular public file: " + name)
        with path.open("rb") as stream:
            data = stream.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise ValueError("public file exceeds size limit: " + name)
        row.update(outcome="read", bytes=len(data))
        return data
    except (OSError, ValueError) as exc:
        row["error"] = type(exc).__name__
        raise
    finally:
        if access is not None:
            access.append(row)


class Organization:
    def __init__(self, root, access=None):
        self.root = Path(root).absolute()
        self.access = access if access is not None else []
        self.inputs = {}
        self.migrations = {}
        pointer = public_path(self.root, CURRENT)
        if pointer.exists():
            body = self.json(CURRENT)
            if not isinstance(body, dict) or type(body.get("schema_version")) is not int or body["schema_version"] != 1 or not isinstance(body.get("inputs"), dict):
                raise ValueError("invalid current organization pointer")
            for key, entry in body["inputs"].items():
                if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                    raise ValueError("invalid current input entry: " + key)
                data = read_public(self.root, entry["path"], access=self.access)
                if hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest() != entry.get("sha256_lf"):
                    raise ValueError("current input version differs: " + key)
                self.inputs[key] = entry["path"]
        document = self.inputs.get("documentation_migration", DOCUMENTS)
        if public_path(self.root, document).exists():
            body = self.json(document)
            if (not isinstance(body, dict) or type(body.get("schema_version")) is not int or body["schema_version"] != 1
                    or not isinstance(body.get("source_commit"), str) or not body["source_commit"] or not isinstance(body.get("entries"), list)):
                raise ValueError("migration requires schema_version=1, source_commit and entries")
            for entry in body["entries"]:
                self.add(entry, document, documentation=True)
        if "archive_locations" in self.inputs:
            source = self.inputs["archive_locations"]
            body = self.json(source)
            if not isinstance(body, dict) or not isinstance(body.get("cumulative_locations"), list):
                raise ValueError("archive requires cumulative_locations")
            for entry in body["cumulative_locations"]:
                if not isinstance(entry, dict) or "current_path" not in entry:
                    raise ValueError("invalid archive entry")
                self.add(dict(entry, new_path=entry["current_path"]), source)
        self.reading = {}
        self.reading_at_target = {}
        if "reading_manifest" in self.inputs:
            body = self.json(self.inputs["reading_manifest"])
            commit = body.get("reading_commit")
            if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
                raise ValueError("reading manifest requires a fixed Git commit")
            for row in body["materials"]:
                old = row["old_path"]
                expected = "https://github.com/LLzijian/grid-transient-rom/blob/" + commit + "/" + old
                if (old not in self.migrations or row["raw_sha256"] != self.migrations[old]["sha256"]
                        or row["reading_url"] != expected):
                    raise ValueError("reading identity differs from verified migration: " + old)
                self.reading[old] = row["reading_url"]
                self.reading_at_target[self.migrations[old]["new_path"]] = row["reading_url"]

    def json(self, name):
        return json.loads(read_public(self.root, name, access=self.access).decode("utf-8-sig"))

    def csv(self, key):
        if key not in self.inputs:
            return []
        data = read_public(self.root, self.inputs[key], access=self.access)
        return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))

    def add(self, entry, source, documentation=False):
        if not isinstance(entry, dict):
            raise ValueError("migration entry must be an object")
        old, new = entry.get("old_path"), entry.get("new_path")
        public_path(self.root, old)
        public_path(self.root, new)
        old_parts, new_parts = PurePosixPath(old).parts, PurePosixPath(new).parts
        if documentation:
            valid = old_parts[0] in {"docs", "reports"} and new_parts[0] in {"reports", "designs"} and old.endswith(".md") and new.endswith(".md")
        else:
            valid = (old_parts[0] in {"reports", "experiments"} and new_parts[0] in {"reports", "experiments", "results"}
                     and PurePosixPath(old).suffix == PurePosixPath(new).suffix)
        if not valid:
            raise ValueError("migration paths violate material scope")
        if old in self.migrations:
            raise ValueError("duplicate migration old_path: " + old)
        if not re.fullmatch(r"[0-9a-fA-F]{64}", str(entry.get("sha256", ""))) or type(entry.get("bytes")) is not int or entry["bytes"] < 0:
            raise ValueError("migration requires sha256 and nonnegative integer bytes")
        data = read_public(self.root, new, kind="documentation" if documentation else "archive", access=self.access)
        if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"].lower():
            raise ValueError("migrated material bytes differ: " + old)
        self.migrations[old] = dict(entry, source_record=source)

    def resolve(self, name):
        path = public_path(self.root, name)
        # An extant original wins; never silently replace an active source.
        if not path.exists() and name in self.migrations:
            return self.migrations[name]["new_path"]
        return name

    def view(self, name=None):
        rows = {r["original_path"]: dict(r) for r in self.csv("scope")}
        if "unreviewed_roles" in self.inputs:
            pending = {r["original_path"] for r in self.csv("unreviewed_roles")}
            for old, row in rows.items():
                row["individual_role_not_reviewed"] = old in pending
        for old, entry in self.migrations.items():
            current = self.resolve(old)
            row = rows.setdefault(old, {})
            row.update(original_path=old, current_path=current, migration_record=entry["source_record"], frozen_original_path=entry["new_path"],
                frozen_sha256=entry["sha256"], continuous_reading=self.reading.get(old))
            if current != old:
                row["handling"] = "archived_byte_identical"
        for entry in self.csv("report_protection"):
            old = entry["original_path"]
            consumers = json.loads(entry["consumers"])
            hint = "execution_or_identity_clue" if any(e["kind"] == "path_hash_binding" or e["source"].endswith((".py", ".ps1")) for e in consumers) else "document_reference_clue"
            rows.setdefault(old, {}).update(original_path=old, current_path=self.resolve(old), protection=dict(
                classification=hint, reviewed=False, runtime_read_verified=None, consumers=consumers,
                release_condition=entry["release_condition"]))
        for entry in self.csv("code_index"):
            old = entry["original_path"]
            rows.setdefault(old, {}).update(original_path=old, current_path=self.resolve(old) if old in self.migrations else entry["current_path"], purpose=entry["title"],
                code_decision=entry["decision"], release_condition=entry["release_condition"])
        if "storage_associations" in self.inputs:
            for entry in self.json(self.inputs["storage_associations"]):
                old = entry["original_path"]
                rows.setdefault(old, {}).update(original_path=old, current_path=entry["path"], storage_association=entry,
                    declared_scope=entry["kind"])
        if "code_protection" in self.inputs:
            for entry in self.json(self.inputs["code_protection"])["indexed_scripts"]:
                old = entry["original_path"]
                if old in rows and entry["consumers"]:
                    rows[old]["protection"] = dict(classification="executable_or_identity_clue", reviewed=False,
                        runtime_read_verified=None, consumers=entry["consumers"], release_condition=entry["release_condition"])
        if "protection_additions" in self.inputs:
            for entry in self.json(self.inputs["protection_additions"]):
                rows[entry["path"]]["protection"] = entry
        if "dependency_review" in self.inputs:
            for entry in self.json(self.inputs["dependency_review"])["items"]:
                rows[entry["original_path"]]["protection"] = entry
        if name is not None:
            canonical(name)
            rows = {k: r for k, r in rows.items() if name in {k, r.get("current_path")}}
            if not rows:
                raise ValueError("no indexed organization record: " + name)
        git = subprocess.run(["git", "ls-files", "-z"], cwd=self.root, capture_output=True)
        tracked = set(git.stdout.decode("utf-8").split("\0")) if git.returncode == 0 else None
        tracked_roots = set(tracked or [])
        for item in tracked or []:
            tracked_roots.update(p.as_posix() for p in PurePosixPath(item).parents)
        for entry in rows.values():
            current = entry["current_path"]
            entry["in_git"] = current in tracked_roots if tracked is not None else None
            entry["git_tracking_scope"] = "path_or_any_descendant"
            entry["local_presence_scope"] = "path_only_not_contents"
            entry["remote_availability"] = "not_verified"
            try:
                path = public_path(self.root, current)
                entry["local_presence"] = "present" if path.exists() else "absent"
            except (OSError, ValueError):
                entry["local_presence"] = "not_inspected"
        return dict(inputs=self.inputs, entries=list(rows.values()), query=name, read_only=True,
                    local_payloads_opened=False, remote_availability_verified=False)


def render_view(value):
    labels = dict(documentation_migration="旧文档迁移", archive_locations="累计归档位置", report_protection="报告保护",
        code_index="代码位置", code_protection="既有代码保护依据", protection_additions="补充保护依据", storage_associations="结果与本地存储",
        scope="文件归属范围", unreviewed_roles="未逐项复核的职责", dependency_review="已复核的依赖样板", reading_manifest="核验过的固定阅读版本")
    lines = ["# 当前整理状态", "", "由current.json的版本化输入生成；历史批次保持原时点。Git跟踪、当前本地存在、远端可取得分别记录；不把public_results名称当作公开可取得的证明。", "",
             "## 当前输入", "", "| 用途 | 来源 |", "|---|---|"]
    lines += [f"| {labels.get(key, key)} | [{path}]({path}) |" for key, path in value["inputs"].items()]
    if value["query"] is None:
        lines += ["", f"已关联{len(value['entries'])}项；按旧路径或当前位置查询：`show-organization --root . --path <路径>`。"]
    else:
        for entry in value["entries"]:
            lines += ["", "## 文件", "", f"原路径：`{entry['original_path']}`", "", f"当前位置：[{entry['current_path']}]({entry['current_path']})", ""]
            if entry.get("frozen_original_path"):
                path = entry["frozen_original_path"]
                lines += [f"冻结原件：[{path}]({path})", ""]
            if entry.get("continuous_reading"):
                lines += [f"[连续阅读]({entry['continuous_reading']})（固定原版本，不作为今天的状态）。", ""]
            lines += ["```json", json.dumps(entry, ensure_ascii=False, indent=2), "```"]
    return "\n".join(lines) + "\n"
