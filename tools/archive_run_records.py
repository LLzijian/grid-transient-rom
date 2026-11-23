"""Create and verify one local ZIP with the records of a closed research Run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
RECORD_SUFFIXES = {".csv", ".json", ".jsonl", ".log", ".md", ".txt"}


def canonical_file(name):
    relative = PurePosixPath(name)
    if (relative.is_absolute() or relative.as_posix() != name or ".." in relative.parts
            or "\\" in name or ":" in name):
        raise ValueError(f"expected a canonical repository path: {name}")
    path = ROOT / name
    if path.resolve() != path or not path.is_file():
        raise ValueError(f"expected a regular repository file: {name}")
    return path


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def bound_sources(run_id, release, names):
    if not release["path"].startswith(f"results/{run_id}/"):
        raise ValueError("expected a release belonging to this Run")
    raw = canonical_file(release["path"]).read_bytes()
    if len(raw) != release["bytes"] or digest(raw) != release["sha256"]:
        raise ValueError("release identity differs")
    body = json.loads(raw)
    if body["run_id"] != run_id:
        raise ValueError("release belongs to another Run")
    for name in names:
        if not name.startswith(("designs/", "src/", "tests/")) or PurePosixPath(name).suffix not in (".md", ".py"):
            raise ValueError(f"expected a design or source reference: {name}")
        if body["source_hashes"].get(name) != digest(canonical_file(name).read_bytes()):
            raise ValueError(f"source differs from the frozen release: {name}")
    return names


def verify(archive, manifest):
    files = {entry["path"]: entry for entry in manifest["files"]}
    with ZipFile(archive) as bundle:
        if (sorted(bundle.namelist()) != sorted([*files, "MANIFEST.json"])
                or bundle.testzip() is not None
                or json.loads(bundle.read("MANIFEST.json")) != manifest):
            raise ValueError(f"archive does not match this Run: {archive}")
        for name, entry in files.items():
            if bundle.getinfo(name).file_size != entry["bytes"] or digest(bundle.read(name)) != entry["sha256"]:
                raise ValueError(f"incorrect ZIP member: {name}")


def archive_run(run_id, local_records):
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id) is None:
        raise ValueError("invalid Run identifier")
    registry = f"harness/runs/{run_id}.json"
    record = json.loads(canonical_file(registry).read_bytes())
    execution = record["execution"]
    if (record["run_id"] != run_id or execution["status"] not in ("COMPLETED", "BLOCKED", "INTERRUPTED_RUNTIME")
            or execution.get("native_live") or execution.get("active_claims")
            or record["scientific_work"]["budget_status"] not in ("closed", "exhausted")):
        raise ValueError(f"Run is not closed: {run_id}")

    scoped = subprocess.check_output(
        ["git", "ls-files", "-z", "--", *(f"{part}/{run_id}/" for part in ("experiments", "reports", "results"))],
        cwd=ROOT,
    ).decode("utf-8").split("\0")
    members = {registry, *(name for name in scoped if name)}
    if "protocol" in record:
        protocol = record["protocol"]
        if not protocol.startswith("campaigns/"):
            raise ValueError("expected the Run protocol in campaigns/")
        members.add(protocol)
    sources = record.get("records_archive_source_refs", [])
    if sources:
        members.update(bound_sources(run_id, execution["release"], sources))
    for name in local_records:
        if not name.startswith(f"artifacts/{run_id}/") or PurePosixPath(name).suffix not in RECORD_SUFFIXES:
            raise ValueError(f"expected a text record belonging to this Run: {name}")
        members.add(name)

    payload = {name: canonical_file(name).read_bytes() for name in sorted(members)}
    manifest = {
        "run_id": run_id,
        "registry_commit": subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--", registry], cwd=ROOT, text=True,
        ).strip() or None,
        "files": [{"path": name, "bytes": len(raw), "sha256": digest(raw)} for name, raw in payload.items()],
    }
    directory = ROOT / "artifacts" / "research_records"
    if directory.resolve() != directory:
        raise ValueError("archive directory must be inside the repository")
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / f"{run_id}_records.zip"
    if archive.resolve() != archive:
        raise ValueError("archive path must not redirect")
    if not archive.exists():
        with TemporaryDirectory(dir=directory) as temporary:
            staged = Path(temporary) / archive.name
            with ZipFile(staged, "x", compression=ZIP_DEFLATED) as bundle:
                for name, raw in payload.items():
                    bundle.writestr(name, raw)
                bundle.writestr("MANIFEST.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
            verify(staged, manifest)
            try:
                archive.hardlink_to(staged)
            except FileExistsError:
                pass
    verify(archive, manifest)
    print(json.dumps({"path": archive.as_posix(), "records": len(payload),
                      "bytes": archive.stat().st_size, "sha256": digest(archive.read_bytes())}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--local-record", action="append", default=[], metavar="REPO_PATH")
    args = parser.parse_args()
    archive_run(args.run_id, args.local_record)
