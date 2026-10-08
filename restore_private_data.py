from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath

BLOCK = 8 * 1024 * 1024
ALLOWED_ROOTS = {"private_data", "external_data"}

class PartReader:
    def __init__(self, directory: Path, names: list[str]):
        self.paths = [directory / n for n in names]
        self.index = 0
        self.current = None

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            raise ValueError("Sequential read requires a size")
        chunks = []
        remaining = n
        while remaining:
            if self.current is None:
                if self.index >= len(self.paths):
                    break
                self.current = self.paths[self.index].open("rb")
                self.index += 1
            block = self.current.read(remaining)
            if not block:
                self.current.close()
                self.current = None
                continue
            chunks.append(block)
            remaining -= len(block)
        return b"".join(chunks)

    def close(self):
        if self.current:
            self.current.close()

def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()

def main():
    parser = argparse.ArgumentParser(description="Verify and restore original dataset and curated result Release volumes")
    parser.add_argument("parts_directory", type=Path)
    parser.add_argument("--verify-only", action="store_true", help="verify every archived file without extracting")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent
    if not args.verify_only:
        existing = [repo / name for name in ALLOWED_ROOTS if (repo / name).exists()]
        if existing:
            raise SystemExit(f"Refusing to overwrite existing directories: {existing}")
    meta = json.loads((repo / "original_research_data_volumes.json").read_text(encoding="utf-8"))
    expected = {}
    for line in (repo / "original_research_data_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        expected[record["path"]] = record
    if len(expected) != meta["source_files"]:
        raise SystemExit("Manifest file count mismatch")

    parts = args.parts_directory.resolve()
    for i, volume in enumerate(meta["volumes"], 1):
        path = parts / volume["name"]
        if not path.is_file() or path.stat().st_size != volume["bytes"]:
            raise SystemExit(f"Missing or wrong-sized volume: {path}")
        if digest_file(path) != volume["sha256"]:
            raise SystemExit(f"SHA-256 mismatch: {path}")
        print(f"verified volume {i}/{len(meta['volumes'])}: {path.name}", flush=True)

    seen = set()
    reader = PartReader(parts, [v["name"] for v in meta["volumes"]])
    try:
        with tarfile.open(fileobj=reader, mode="r|") as archive:
            for member in archive:
                name = member.name
                path_parts = PurePosixPath(name).parts
                if (not member.isfile() or len(path_parts) < 2
                        or path_parts[0] not in ALLOWED_ROOTS
                        or any(p in ("", ".", "..") for p in path_parts)):
                    raise SystemExit(f"Unexpected tar member: {name}")
                if name not in expected or name in seen:
                    raise SystemExit(f"Unexpected or duplicate file: {name}")
                record = expected[name]
                if member.size != record["bytes"]:
                    raise SystemExit(f"Size mismatch: {name}")
                digest = hashlib.sha256()
                out = None
                if not args.verify_only:
                    destination = repo.joinpath(*path_parts)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    out = destination.open("xb")
                try:
                    source = archive.extractfile(member)
                    if source is None:
                        raise SystemExit(f"Cannot read: {name}")
                    with source:
                        while True:
                            block = source.read(BLOCK)
                            if not block:
                                break
                            digest.update(block)
                            if out:
                                out.write(block)
                finally:
                    if out:
                        out.close()
                if digest.hexdigest() != record["sha256"]:
                    raise SystemExit(f"File SHA-256 mismatch: {name}")
                seen.add(name)
                if len(seen) % 200 == 0:
                    print(f"verified {len(seen)}/{len(expected)} files", flush=True)
    finally:
        reader.close()
    if seen != set(expected):
        raise SystemExit(f"Missing {len(set(expected) - seen)} files")
    print(f"{'Verified' if args.verify_only else 'Restored'} {len(seen)} files successfully", flush=True)

if __name__ == "__main__":
    main()
