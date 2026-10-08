from __future__ import annotations

import copy
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# The regression tests exercise path/selection guards only; load the module
# under a private alias with a temporary ANDES stub and restore import state.
_sentinel = object()
_previous_andes = sys.modules.get("andes", _sentinel)
try:
    if _previous_andes is _sentinel:
        sys.modules["andes"] = types.ModuleType("andes")
    _spec = importlib.util.spec_from_file_location(
        "_boundary_generate_private_dataset_0075",
        ROOT / "experiments/generate_private_dataset.py",
    )
    assert _spec is not None and _spec.loader is not None
    generator = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(generator)
finally:
    if _previous_andes is _sentinel:
        sys.modules.pop("andes", None)
    else:
        sys.modules["andes"] = _previous_andes

def _config(name: str = "configs/private_dataset_boundary_0075.json"):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def test_boundary_explicit_selection_is_sorted_and_train_only():
    config = _config()
    events = generator._boundary_events(config)
    assert len(events) == 20
    assert [event["bus"] for event in events] == sorted(event["bus"] for event in events)
    assert not ({event["bus"] for event in events} &
                (set(config["validation_buses"]) | set(config["test_buses"])))
    assert len({(event["line"], event["location"], event["bus"], event["xf"], event["duration_s"])
                for event in events}) == 20


def test_boundary_selection_rejects_heldout_and_duplicate():
    config = _config()
    heldout = copy.deepcopy(config)
    heldout["smoke"][0]["bus"] = heldout["test_buses"][0]
    with pytest.raises(ValueError, match="held-out"):
        generator._boundary_events(heldout)
    duplicate = copy.deepcopy(config)
    duplicate["smoke"][1] = copy.deepcopy(duplicate["smoke"][0])
    with pytest.raises(ValueError, match="duplicate"):
        generator._boundary_events(duplicate)


def test_default_solver_load_kwargs_preserve_old_path():
    kwargs, workdir = generator._solver_load_kwargs(None)
    assert kwargs == {"setup": False, "no_output": True, "default_config": True}
    assert workdir is None


def test_workspace_solver_load_kwargs_are_explicit_and_reusable():
    workdir = ROOT / ".loop-work" / "boundary_codegen_test"
    kwargs, returned = generator._solver_load_kwargs(workdir)
    assert returned == workdir
    assert kwargs["setup"] is False and kwargs["no_output"] is True and kwargs["default_config"] is True
    assert kwargs["pycode_path"] == str(workdir / "andes_pycode")
    assert kwargs["output_path"] == str(workdir / "andes_output")
    assert kwargs["no_undill"] is True and kwargs["autogen_stale"] is False


def test_solver_workdir_must_be_workspace_local_and_nonroot():
    assert generator._resolve_solver_workdir(ROOT / ".loop-work" / "boundary_codegen_test")
    with pytest.raises(ValueError):
        generator._resolve_solver_workdir(ROOT)
    with pytest.raises(ValueError):
        generator._resolve_solver_workdir(ROOT / "configs")


def test_fine_config_changes_only_integration_step_and_note():
    coarse = _config("configs/private_dataset_boundary_0075.json")
    fine = _config("configs/private_dataset_boundary_fine_0075.json")
    assert fine["integration_step_s"] == 0.0025
    assert coarse["integration_step_s"] == 0.005
    for key in coarse:
        if key not in {"integration_step_s", "notes"}:
            assert coarse[key] == fine[key], key
