from __future__ import annotations

import json
from pathlib import Path

import pytest

from kiki.models.prunable_energy_model import PrunableEnergyModel

ROOT = Path(__file__).resolve().parents[1]


def test_module_adoption_threshold() -> None:
    accepted = PrunableEnergyModel.module_adoption_decision(
        {"standardized_loss": 10.0, "failure_rate": 0.0},
        {"standardized_loss": 9.7, "failure_rate": 0.0},
    )
    rejected = PrunableEnergyModel.module_adoption_decision(
        {"standardized_loss": 10.0, "failure_rate": 0.0},
        {"standardized_loss": 9.9, "failure_rate": 0.0},
    )
    failed = PrunableEnergyModel.module_adoption_decision(
        {"standardized_loss": 10.0, "failure_rate": 0.0},
        {"standardized_loss": 9.0, "failure_rate": 0.1},
    )
    assert accepted["accepted"] is True
    assert rejected["accepted"] is False
    assert failed["accepted"] is False


def _legacy_fit_loss(path: str) -> float:
    payload = json.loads((ROOT / path).read_text(encoding="utf-8"))
    return float(payload["fit"]["best_validation_loss"])


def test_legacy_f1_accepts_and_f4_rejects_total_module_gate() -> None:
    f1 = PrunableEnergyModel.module_adoption_decision(
        {"standardized_loss": _legacy_fit_loss("results/e02_backbone/summary.json")},
        {"standardized_loss": _legacy_fit_loss("results/e02_full/summary.json")},
    )
    f4 = PrunableEnergyModel.module_adoption_decision(
        {
            "standardized_loss": _legacy_fit_loss(
                "results/e06/F4/backbone/summary.json"
            )
        },
        {"standardized_loss": _legacy_fit_loss("results/e06/F4/full/summary.json")},
    )
    assert f1["accepted"] is True
    assert f4["accepted"] is False


def test_model_round_trip_preserves_validation_scales(tmp_path: Path) -> None:
    model = PrunableEnergyModel.load(
        ROOT / "results" / "e02_backbone" / "model_base.npz"
    )
    model.validation_scales = {"omega": 2.5, "edge_angle": 0.25}
    path = tmp_path / "round_trip.npz"
    model.save(path)

    restored = PrunableEnergyModel.load(path)

    assert restored.validation_scales == model.validation_scales


def test_model_save_rejects_invalid_validation_scales(tmp_path: Path) -> None:
    model = PrunableEnergyModel.load(
        ROOT / "results" / "e02_backbone" / "model_base.npz"
    )
    model.validation_scales = {"omega": 1.0, "edge_angle": 0.0}

    with pytest.raises(ValueError, match="positive finite"):
        model.save(tmp_path / "invalid.npz")
