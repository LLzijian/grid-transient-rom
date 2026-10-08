from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"


def _loads_payload_inside_test_id_loop(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for loop in (node for node in ast.walk(tree) if isinstance(node, ast.For)):
        if "test_ids" not in ast.dump(loop.iter, include_attributes=False):
            continue
        for node in ast.walk(loop):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr in {"load_event", "load_event_once"}:
                return True
    return False


def test_only_release_entry_point_reads_test_payloads() -> None:
    readers = {
        path.name
        for path in EXPERIMENTS.glob("*.py")
        if _loads_payload_inside_test_id_loop(path)
    }
    assert readers == {"run_test_release.py"}


def test_legacy_reopen_and_modal_entry_points_are_removed() -> None:
    removed = {
        "run_e5.py",
        "run_e5b.py",
        "run_e5c.py",
        "run_e6b_test.py",
        "run_e6c.py",
        "run_e8.py",
    }
    assert not {name for name in removed if (EXPERIMENTS / name).exists()}

    modal_source = (EXPERIMENTS / "run_modal_validation.py").read_text(
        encoding="utf-8"
    )
    assert "model_pruned.npz" in modal_source
    assert "model_full.npz" not in modal_source
    assert "per_event_best" not in modal_source
    assert "best_light" not in modal_source
