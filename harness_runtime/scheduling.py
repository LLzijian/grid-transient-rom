"""Deterministic, advisory scheduling of a supplied task snapshot.

Input order is the explicit selection order. Optional ``priority`` is metadata;
the host may order tasks before this call. This is neither an optimal scheduler
nor an agent dispatcher. Resource capacities and agent slots come from the host.
Paths use repository-relative POSIX syntax and are checked lexically: an actual
executor must also enforce its filesystem boundary, including symlinks.
"""

from collections import deque
from pathlib import PurePosixPath
import os
import re


STATUSES = {"pending", "running", "completed", "failed", "blocked", "cancelled", "paused"}
REQUIRED_FIELDS = {
    "task_id", "role", "question", "inputs", "outputs", "write_paths",
    "depends_on", "resources", "status", "acceptance", "stop_condition",
}


def _quantity(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def normalize_write_path(value):
    """Return one safe lexical path; parent traversal and glob syntax are rejected."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("write path must be a nonempty, unpadded string")
    if any(ord(char) < 32 for char in value) or "\\" in value:
        raise ValueError("write path must use POSIX separators without control characters")
    if value.startswith("/") or re.match(r"^[A-Za-z]:", value) or ":" in value:
        raise ValueError("write path must be repository-relative")
    if any(char in value for char in "*?[]{}"):
        raise ValueError("write path cannot contain wildcard syntax")
    if ".." in value.split("/"):
        raise ValueError("write path cannot contain parent traversal")
    normalized = str(PurePosixPath(value))
    if normalized == ".":
        raise ValueError("declare a bounded write path, not the entire repository")
    # Keep contracts portable to Windows: these spellings can alias ordinary
    # files or name devices, even when the supplied separators are POSIX-style.
    for part in PurePosixPath(normalized).parts:
        if part.endswith((".", " ")) or any(char in part for char in '<>"|'):
            raise ValueError("write path cannot contain Windows aliases or invalid characters")
        stem = part.split(".", 1)[0].rstrip(" ").upper()
        if stem in {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"} or re.fullmatch(r"(?:COM|LPT)[1-9¹²³]", stem):
            raise ValueError("write path cannot name a Windows reserved device")
    return normalized


def paths_overlap(left, right):
    """Directory-prefix overlap, without confusing e.g. src with src2."""
    if os.name == "nt":
        # Be conservative even if an individual Windows directory opts into
        # case-sensitive names; the scheduler does not inspect the filesystem.
        left, right = left.casefold(), right.casefold()
    return left == right or left.startswith(right + "/") or right.startswith(left + "/")


def ready_tasks(tasks, available_resources, max_agent_slots):
    """Validate a DAG and select a feasible ready group without changing inputs.

    Completed dependencies alone satisfy a dependency. Running tasks occupy one
    host slot each, their resources, and their write paths. Pending tasks are
    considered once, in input order. Read-only tasks have empty ``write_paths``;
    they may run together. Inputs should be immutable snapshots or protected by
    explicit dependencies; this function does not infer read/write hazards.

    Returns ``valid``, typed ``issues``, task IDs in ``ready`` and ``running``,
    per-task ``blocked`` reasons, and the remaining resource/slot accounting.
    Any malformed DAG or inconsistent active allocation prevents a ready group.
    No task, job, process, agent, file write, or model call is started here.
    """
    issues = []

    def issue(code, message, task_id=None):
        row = {"severity": "error", "code": code, "message": message}
        if task_id is not None:
            row["task_id"] = task_id
        issues.append(row)

    result = {
        "valid": False, "issues": issues, "ready": [], "running": [], "blocked": {},
        "advisory_only": True, "dispatch_performed": False,
        "authorization": "not_granted; ready means dependency/resource-feasible planning candidates only",
        "input_hashes_verified": False,
        "selection_rule": "input_order_greedy; optional priority is metadata only",
    }
    if not isinstance(tasks, list):
        issue("invalid_tasks", "tasks must be a list")
        return result
    if not isinstance(available_resources, dict):
        issue("invalid_resources", "available_resources must be a dictionary")
        return result
    if not _quantity(max_agent_slots):
        issue("invalid_slots", "max_agent_slots must be a nonnegative integer supplied by the host")
    for name, quantity in available_resources.items():
        if not isinstance(name, str) or not name.strip() or not _quantity(quantity):
            issue("invalid_resource_capacity", "resource names must be nonempty strings and quantities nonnegative integers")

    normalized = []
    ids = set()
    for ordinal, raw in enumerate(tasks):
        if not isinstance(raw, dict):
            issue("invalid_task", f"task at index {ordinal} must be a dictionary")
            continue
        task_id = raw.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            issue("invalid_task_id", f"task at index {ordinal} needs a nonempty task_id")
            continue
        if task_id in ids:
            issue("duplicate_task_id", "task IDs must be unique", task_id)
        ids.add(task_id)
        missing = REQUIRED_FIELDS - raw.keys()
        if missing:
            issue("missing_fields", "missing fields: " + ", ".join(sorted(missing)), task_id)
        for field in ("role", "question", "stop_condition"):
            if not isinstance(raw.get(field), str) or not raw[field].strip():
                issue("invalid_task_field", f"{field} must be nonempty text", task_id)
        acceptance = raw.get("acceptance")
        if not ((isinstance(acceptance, str) and acceptance.strip()) or
                (isinstance(acceptance, list) and acceptance and
                 all(isinstance(x, str) and x.strip() for x in acceptance))):
            issue("invalid_acceptance", "acceptance must be nonempty text or a nonempty list of text", task_id)
        inputs = raw.get("inputs")
        if not isinstance(inputs, list) or not all(
                (isinstance(x, str) and x.strip()) or
                (isinstance(x, dict) and isinstance(x.get("path"), str) and x["path"].strip()
                 and all(isinstance(x[k], str) and x[k].strip() for k in ("sha256", "commit") if k in x))
                for x in inputs):
            issue("invalid_task_field", "inputs must be a list of text or {path, optional sha256/commit} records; hashes are verified by the host", task_id)
        for field in ("outputs", "depends_on", "write_paths"):
            values = raw.get(field)
            if not isinstance(values, list) or not all(isinstance(x, str) and x.strip() for x in values):
                issue("invalid_task_field", f"{field} must be a list of nonempty strings", task_id)
        resources = raw.get("resources")
        if not isinstance(resources, dict):
            issue("invalid_task_resources", "task resources must be a dictionary", task_id)
            resources = {}
        for name, quantity in resources.items():
            if not isinstance(name, str) or not name.strip() or not _quantity(quantity):
                issue("invalid_task_resources", "resource names must be nonempty strings and quantities nonnegative integers", task_id)
        status = raw.get("status")
        if not isinstance(status, str) or status not in STATUSES:
            issue("invalid_status", "status must be one of " + ", ".join(sorted(STATUSES)), task_id)
        if "priority" in raw and (not isinstance(raw["priority"], int) or isinstance(raw["priority"], bool)):
            issue("invalid_priority", "optional priority must be an integer; input order still controls selection", task_id)
        paths = []
        for path in raw.get("write_paths", []) if isinstance(raw.get("write_paths"), list) else []:
            try:
                paths.append(normalize_write_path(path))
            except ValueError as exc:
                issue("invalid_write_path", str(exc), task_id)
        if "read_only" in raw and not isinstance(raw["read_only"], bool):
            issue("invalid_read_only", "read_only, when present, must be boolean", task_id)
        if raw.get("read_only") is True and paths:
            issue("read_only_has_writes", "a read-only task cannot own write paths", task_id)
        if raw.get("read_only") is False and (not paths or not raw.get("outputs")):
            issue("missing_write_contract", "an explicit implementation task needs write_paths and outputs", task_id)
        dependencies = raw.get("depends_on", [])
        normalized.append({
            "task_id": task_id, "status": status, "resources": dict(resources),
            "write_paths": list(dict.fromkeys(paths)),
            "depends_on": dependencies if isinstance(dependencies, list) and all(isinstance(x, str) for x in dependencies) else [],
        })

    if issues:
        result["running"] = [t["task_id"] for t in normalized if t["status"] == "running"]
        return result

    by_id = {t["task_id"]: t for t in normalized}
    indegree = {t["task_id"]: 0 for t in normalized}
    children = {t["task_id"]: [] for t in normalized}
    for task in normalized:
        for dependency in dict.fromkeys(task["depends_on"]):
            if dependency not in by_id:
                issue("missing_dependency", f"unknown dependency: {dependency}", task["task_id"])
                continue
            indegree[task["task_id"]] += 1
            children[dependency].append(task["task_id"])
    queue = deque(task_id for task_id, count in indegree.items() if count == 0)
    visited = 0
    while queue:
        task_id = queue.popleft()
        visited += 1
        for child in children[task_id]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if visited != len(normalized):
        issue("dependency_cycle", "dependency cycle or descendants of a cycle: " + ", ".join(task_id for task_id, n in indegree.items() if n))

    running = [t for t in normalized if t["status"] == "running"]
    result["running"] = [t["task_id"] for t in running]
    occupied = []
    used = {name: 0 for name in available_resources}
    for task in running:
        for dependency in task["depends_on"]:
            if dependency in by_id and by_id[dependency]["status"] != "completed":
                issue("active_dependency_incomplete", f"running task dependency is not completed: {dependency}", task["task_id"])
        for name, quantity in task["resources"].items():
            used[name] = used.get(name, 0) + quantity
        for path in task["write_paths"]:
            for other_id, other_path in occupied:
                if other_id != task["task_id"] and paths_overlap(path, other_path):
                    issue("active_write_conflict", f"running write path {path} overlaps {other_id}:{other_path}", task["task_id"])
            occupied.append((task["task_id"], path))
    for name, quantity in used.items():
        if quantity > available_resources.get(name, 0):
            issue("active_resource_over_capacity", f"running tasks use {quantity} units of {name}, host capacity is {available_resources.get(name, 0)}")
    if len(running) > max_agent_slots:
        issue("active_slots_over_capacity", f"{len(running)} running tasks exceed host slot count {max_agent_slots}; no work is cancelled")
    remaining = {name: available_resources.get(name, 0) - used.get(name, 0) for name in available_resources.keys() | used.keys()}
    slots_left = max_agent_slots - len(running)
    result["resources"] = {"capacity": dict(available_resources), "running_usage": used, "remaining_after_selection": remaining}
    result["slots"] = {"capacity": max_agent_slots, "running": len(running), "remaining_after_selection": slots_left}
    if issues:
        return result

    for task in normalized:
        task_id = task["task_id"]
        if task["status"] != "pending":
            if task["status"] not in {"running", "completed"}:
                result["blocked"][task_id] = ["status:" + task["status"]]
            continue
        reasons = []
        for dependency in task["depends_on"]:
            if by_id[dependency]["status"] != "completed":
                reasons.append("dependency_not_completed:" + dependency)
        if slots_left <= 0:
            reasons.append("host_slots_unavailable")
        for name, quantity in task["resources"].items():
            if quantity > remaining.get(name, 0):
                reasons.append("resource_unavailable:" + name)
        for path in task["write_paths"]:
            for other_id, other_path in occupied:
                if paths_overlap(path, other_path):
                    reasons.append("write_conflict:" + other_id + ":" + other_path)
        if reasons:
            result["blocked"][task_id] = list(dict.fromkeys(reasons))
            continue
        result["ready"].append(task_id)
        slots_left -= 1
        for name, quantity in task["resources"].items():
            remaining[name] = remaining.get(name, 0) - quantity
        occupied.extend((task_id, path) for path in task["write_paths"])
    result["slots"]["remaining_after_selection"] = slots_left
    result["valid"] = True
    return result
