"""Strict JSON identity adapters for the unchanged closure checkpoint helpers."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def _snapshot_value(value, active):
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise TypeError("checkpoint identity rejects non-finite floats")
        return value
    if type(value) in (list, tuple):
        ident = id(value)
        if ident in active:
            raise TypeError("checkpoint identity rejects cyclic containers")
        active.add(ident)
        try:
            return [_snapshot_value(item, active) for item in value]
        finally:
            active.remove(ident)
    if type(value) is dict:
        ident = id(value)
        if ident in active:
            raise TypeError("checkpoint identity rejects cyclic containers")
        active.add(ident)
        try:
            result = {}
            for key, item in value.items():
                if type(key) is not str:
                    raise TypeError("checkpoint identity requires string dictionary keys")
                result[key] = _snapshot_value(item, active)
            return result
        finally:
            active.remove(ident)
    raise TypeError(f"unsupported checkpoint identity type: {type(value).__name__}")


def identity_snapshot(identity):
    """Return a detached JSON-compatible identity; tuples become ordered lists."""
    return _snapshot_value(identity, set())


def _canonical(snapshot):
    # Validate caller-supplied snapshots too; do not allow mutable/non-JSON values.
    clean = identity_snapshot(snapshot)
    return json.dumps(clean, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def identity_fingerprint(snapshot):
    """Typed canonical fingerprint; JSON preserves bool/int/float distinctions."""
    return hashlib.sha256(_canonical(snapshot)).hexdigest()


def _checked_metadata(output):
    root = Path(output).resolve()
    pointer = json.loads((root / "latest.json").read_text(encoding="utf-8"))
    directory = pointer["directory"]
    if (type(directory) is not str or not directory or directory in (".", "..") or
            Path(directory).name != directory or "/" in directory or "\\" in directory):
        raise ValueError("checkpoint directory must be an immediate child")
    folder = root / directory
    if folder.resolve().parent != root:
        raise ValueError("checkpoint directory must resolve to an immediate child")
    metadata_path = folder / "checkpoint.json"
    digest = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    if digest != pointer["checkpoint_sha256"]:
        raise ValueError("checkpoint metadata hash changed")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def save_identity_checkpoint(save_checkpoint, output, model, opt, rng, coverage,
                             step, identity, trace):
    """Snapshot once, then pass that exact JSON value to the legacy writer."""
    snapshot = identity_snapshot(identity)
    _canonical(snapshot)
    return save_checkpoint(output, model, opt, rng, coverage, step, snapshot, trace)


def restore_identity_checkpoint(restore_checkpoint, output, device, expected_identity,
                                allow_device_change=False):
    """Typed-check identity before delegating all legacy integrity checks/restoration."""
    snapshot = identity_snapshot(expected_identity)
    expected_fingerprint = identity_fingerprint(snapshot)
    metadata = _checked_metadata(output)
    saved = metadata["identity"]
    if identity_fingerprint(saved) != expected_fingerprint:
        raise ValueError("typed source, train, configuration or code identity changed")
    # The exact same explicit snapshot is supplied to the unchanged legacy helper.
    return restore_checkpoint(output, device, snapshot, allow_device_change)
