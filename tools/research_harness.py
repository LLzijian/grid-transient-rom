#!/usr/bin/env python3
"""Offline record checks and planning; never dispatch agents or execute science.

Examples (paths are supplied explicitly):
  python tools/research_harness.py assess --record harness/examples/run_single_matched_gain.json
  python tools/research_harness.py schedule --tasks harness/examples/tasks.json --resources '{"gpu:0":1}' --slots 3
  python tools/research_harness.py render-state --registry harness/examples/registry.snapshot.json
  python tools/research_harness.py check --root .

All commands are read-only unless render-state receives an explicit --output.
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
PRIVATE_PARTS = {"private_data", "private-data", "datasets", "checkpoints", "weights", ".git"}
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
    jobs = registry.get("active_jobs", [])
    pointers = registry.get("pointers", {})
    if not isinstance(snapshot, dict) or not isinstance(current, dict) or not isinstance(jobs, list) or not isinstance(pointers, dict):
        raise ValueError("snapshot/current_run/pointers must be objects and active_jobs a list")

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
        f"The snapshot records {len(jobs)} active job(s). Live process state was not inspected.", "",
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
        elif path.suffix.lower() in {".json", ".md"}:
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
        if not _inside(target, root):
            issue("reference_outside_root", "reference escapes the repository: " + ref, source)
            return
        if _private(target.relative_to(root)):
            issue("private_reference_not_read", "opaque private reference was not opened or checked: " + ref, source, "warning")
            return
        refs.append({"source": source.relative_to(root).as_posix(), "reference": ref})
        if not target.exists():
            bundled = target.relative_to(root).parts[:1] in [("harness",), ("harness_runtime",), ("tools",), ("templates",)]
            severity = "warning" if package_mode and not bundled else "error"
            issue("missing_reference", "relative reference is absent: " + ref, source, severity)

    files = set()
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
                        if target.suffix in {".json", ".md", ".py"}:
                            files.add(target)
        except (OSError, ValueError) as exc:
            issue("invalid_manifest", str(exc), manifest)

    for path in sorted(files):
        if not _safe_public_path(path, root):
            issue("unsafe_metadata_path", "public metadata path is outside the allowed scope", path)
            continue
        checked.append(path.relative_to(root).as_posix())
        try:
            content = path.read_text(encoding="utf-8")
            if path.suffix == ".json":
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
            elif path.suffix == ".md":
                for ref in LINK_PATTERN.findall(content):
                    reference(ref, path)
            elif path.suffix == ".py":
                ast.parse(content, filename=str(path))
        except (OSError, UnicodeError, ValueError, SyntaxError) as exc:
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
    render.add_argument("--output", help="explicit destination; this is the only writing operation")
    check = commands.add_parser("check", help="check public package metadata without an installed active registry")
    check.add_argument("--root", required=True)
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
        else:
            result = check_root(args.root)
        emit(result)
        return 0 if result.get("valid") is True else 1
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        emit({"valid": False, "issues": [{"severity": "error", "code": "input_error", "message": str(exc)}], "dispatch_performed": False})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
