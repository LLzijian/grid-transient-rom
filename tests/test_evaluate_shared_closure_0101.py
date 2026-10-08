"""Synthetic evaluator-boundary tests; no private data, rollout, or TEST access."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import evaluate_shared_closure_0101 as evaluator


def test_f_power_is_explicitly_not_applicable_without_touching_hamiltonian(monkeypatch):
    class FModel:
        @property
        def energy(self):
            raise AssertionError("ordinary F evaluator must not probe a fake energy")

    monkeypatch.setattr("evaluate_local_memory_0098.energy_power_qa",
                        lambda model: (_ for _ in ()).throw(AssertionError("must not call")),
                        raising=False)
    result = evaluator._energy_power_result(FModel(), "F")
    assert result["status"] == "not_applicable"
    assert result["pass"] is None
    assert "no Hamiltonian" in result["reason"]


@pytest.mark.parametrize(("model_kind", "arm", "family"), [
    ("S23", "L10-S", "S"), ("F23", "L10-S", "F"),
])
def test_model_family_uses_explicit_kind_and_checks_checkpoint_metadata(tmp_path, model_kind, arm, family):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "checkpoint.json").write_text(json.dumps({"model_kind": model_kind, "arm": arm}))

    assert evaluator._model_family(checkpoint, SimpleNamespace(model_kind=model_kind, arm=arm)) == family


def test_model_family_rejects_metadata_mismatch(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "checkpoint.json").write_text(json.dumps({"model_kind": "S23", "arm": "F23"}))

    class Model:
        model_kind = "S23"
        arm = "L10-S"

    with pytest.raises(ValueError, match="arm says F23"):
        evaluator._model_family(checkpoint, Model())


def test_checkpoint_directory_hashes_numeric_export_without_assuming_raw_model_pt(tmp_path):
    export = tmp_path / "structured_numeric_export.json"
    export.write_text('{"format":"0101"}')
    (tmp_path / "model.pt").write_bytes(b"opaque raw state")
    files = evaluator._checkpoint_paths(tmp_path)
    assert "checkpoint/structured_numeric_export.json" in files
    assert "checkpoint/model.pt" in files


def test_cli_runner_rejects_caps_above_registered_max_before_spawning(tmp_path):
    with pytest.raises(ValueError, match="cap_seconds"):
        evaluator.run(tmp_path / "missing", tmp_path / "out", cap_seconds=750.01)


def test_registered_cohort_guard_keeps_train_and_validation_exclusions_separate():
    train = [{"id": f"t{i}", "group": f"g{i % 24}",
              "label": "unstable" if i < 35 else "stable"} for i in range(346)]
    validation = [{"id": f"v{i}", "group": f"g{i % 24}",
                   "label": "unstable" if i < 12 else "stable"} for i in range(107)]
    excluded = {**{f"te{i}": "train" for i in range(23)},
                **{f"ve{i}": "validation" for i in range(10)}}
    evaluator._check_cohorts(train, excluded, validation, [{} for _ in range(10)])
