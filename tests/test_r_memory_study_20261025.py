"""Pure study-contract checks: no real dataset or optimizer updates."""
from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src")]
import r_memory_study_20261025 as study


def events():
    result = []
    for group in range(20):
        for index in range(3):
            result.append({"id": f"event_{group:02d}_{index}", "group": str(group),
                "label": "unstable" if index == 0 else "stable",
                "x": np.zeros((19 + index, 30)), "torque": np.zeros((19 + index, 10))})
    return result


def test_whole_group_partition_is_fixed_and_source_exposure_is_not_erased():
    source = events()
    fit, held = study.partition_events(source)
    assert {e["id"] for e in fit} | {e["id"] for e in held} == {e["id"] for e in source}
    assert not {e["group"] for e in fit} & {e["group"] for e in held}
    assert study.partition_events(list(reversed(source)))[0][0]["id"] == fit[0]["id"]
    for event in held:
        integer = int.from_bytes(hashlib.sha256(("r_memory_20261025|" + event["group"]).encode()).digest(), "big")
        assert integer % 5 == 0
    with pytest.raises(ValueError, match="empty or full"):
        study.partition_events([source[0]])


def test_calibration_panel_retains_both_outcomes_and_group_round_robin():
    fit, _ = study.partition_events(events())
    panel = study.panel_events(fit)
    assert len(panel) == len({e["id"] for e in panel}) == 12
    assert sum(e["label"] == "unstable" for e in panel) == 3
    assert sum(e["label"] == "stable" for e in panel) == 9
    assert len({e["group"] for e in panel if e["label"] == "unstable"}) == 3


def test_frozen_schedule_pairs_models_and_uses_real_ragged_counts():
    fit, _ = study.partition_events(events())
    first = study.make_schedule(fit, 25101)
    second = study.make_schedule(fit, 25101)
    assert first == second
    rows = first["rows"]
    assert len(rows) == 2048
    assert [rows[i - 1]["horizon_s"] for i in (1, 256, 257, 512, 513, 514, 2048)] == [1., 1., 2., 2., 5., 8., 8.]
    by_id = {e["id"]: e for e in fit}
    for row in rows:
        assert len(row["sample_ids"]) == 12
        assert row["long_counts"] == [len(by_id[eid]["x"]) - 11 for eid in row["sample_ids"]]
        assert all(10 <= endpoint < len(by_id[eid]["x"]) - 1
            for eid, endpoint in zip(row["sample_ids"], row["endpoints"]))
    with pytest.raises(ValueError):
        study.horizon_for(2049)


def test_calibration_joint_choice_is_arithmetic_and_requires_both_valid():
    scores = {"A": {"1": {"qualified": True, "ratio": .7}, "3": {"qualified": True, "ratio": .8}},
              "B": {"1": {"qualified": True, "ratio": .1}, "3": {"qualified": True, "ratio": 1.}},
              "C": {"1": {"qualified": True, "ratio": 3.}, "3": {"qualified": True, "ratio": 1.}}}
    assert study.select_multipliers(scores) == {"A": 1, "B": 3, "C": 3}
    scores["B"]["3"]["qualified"] = False
    assert study.select_multipliers(scores)["B"] == 1
    scores["C"]["1"]["qualified"] = False
    assert study.select_multipliers(scores)["C"] is None
    scores["A"]["3"]["ratio"] = .7
    assert study.select_multipliers(scores)["A"] == 1


def test_noise_is_event_paired_and_preserves_all_p_and_future_truth():
    event = events()[0]
    event["x"] = np.arange(19 * 30, dtype=float).reshape(19, 30) * .001
    prior = event["x"].copy()
    first = study.noise_history(event, np.arange(1, 11), np.ones(10) * 2.)
    second = study.noise_history(event, np.arange(1, 11), np.ones(10) * 2.)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[:, 20:30], event["x"][:11, 20:30])
    np.testing.assert_array_equal(event["x"], prior)
    assert np.any(first[-1, :20] != event["x"][10, :20])
    different = study.noise_history({**event, "id": "different_event"}, np.arange(1, 11), np.ones(10) * 2.)
    assert not np.array_equal(first[:, :20], different[:, :20])


def test_latent_normalization_distinguishes_old_q_and_physical_r():
    class Model:
        mass = torch.ones(10, dtype=torch.float64)
        rstar = torch.arange(1., 11., dtype=torch.float64)
    model = Model()
    torch.testing.assert_close(study.latent_scale(model, "A"), torch.ones(10, dtype=torch.float64) * .1)
    torch.testing.assert_close(study.latent_scale(model, "B"), model.rstar * .1)
    model.rstar = torch.zeros(10)
    with pytest.raises(ValueError, match="positive physical rstar"):
        study.latent_scale(model, "C")


def test_fit_only_cache_does_not_materialize_no_update_payload(tmp_path, monkeypatch):
    fit, held = study.partition_events(events())
    study._save_event_cache(tmp_path / "fit_events.npz", fit)
    study._save_event_cache(tmp_path / "no_update_events.npz", held)
    study.write(tmp_path / "schedule.json", {})
    metadata = {"source_sha256": study.SOURCE_SHA, "code": {"fixture": "pure"},
        "schedule_sha256": study.sha(tmp_path / "schedule.json"),
        "fit_cache_sha256": study.sha(tmp_path / "fit_events.npz"),
        "no_update_cache_sha256": study.sha(tmp_path / "no_update_events.npz"),
        "fit_ids": [e["id"] for e in fit], "no_update_ids": [e["id"] for e in held]}
    study.write(tmp_path / "prepare.json", metadata)
    (tmp_path / "no_update_events.npz").write_bytes(b"deliberately unreadable held payload")
    monkeypatch.setattr(study, "require_source", lambda: None)
    monkeypatch.setattr(study, "code_identity", lambda: {"fixture": "pure"})
    _, restored_fit = study.checked_preparation(tmp_path, fit_only=True)
    assert [e["id"] for e in restored_fit] == metadata["fit_ids"]
    with pytest.raises(ValueError, match="cache identity"):
        study.checked_preparation(tmp_path, fit_only=False)


def test_eight_second_gate_controls_primary_five_but_fifteen_failure_does_not(tmp_path, monkeypatch):
    import summarize_train_local_memory_0098
    monkeypatch.setattr(summarize_train_local_memory_0098, "build_summary", lambda _: {})
    rows = [{"event_id": "e", "group": "g", "label": "stable", "horizon": h,
             "available": True, "finite": True, "numerically_admissible": True,
             "mean_point_error": value} for h, value in (("5", 1.), ("8", 1.5))]
    records = {key: [] for key in ("records", "common_frequency", "normalized_p", "numerical")}
    records["objective_point_error"] = rows
    pair = {"finite_to_endpoint": True, "los_pair_pass": True,
        "coi_normalized_rmse": 0., "common_normalized_rmse": 0.,
        "fine_classification": {"predicted_los": False, "window_settled": True, "trigger_index": None}}
    extra = {"event_id": "e", "label": "stable", "scientific_numeric_8s": False,
        "classification_8s": copy.deepcopy(pair), "classification_15s": copy.deepcopy(pair),
        "coarse_point_errors": {"5": .9, "8": 1.4},
        "actual_interval15": {"available": True, "finite": False, "numerically_admissible": False}}
    blocked = study.group_summary(records, [extra], {"e"})
    assert blocked["primary"]["value"] is None
    assert blocked["horizon_risks"]["5"]["value"] is None
    extra["scientific_numeric_8s"] = True
    retained = study.group_summary(records, [extra], {"e"})
    assert retained["primary"]["value"] == 1.
    assert retained["horizon_risks"]["8"]["value"] == pytest.approx(1.5)
    assert retained["horizon_risks"]["5"]["numerical_score_perturbation"] == pytest.approx(.1)
    assert retained["all_8s_rows_qualified"]
    assert not retained["all_actual15s_rows_qualified"]


def endpoint_fixture(base, monkeypatch):
    """Pure file identities and a fake evaluator; no source model/data loading."""
    study.write(base / "prepare.json", {"pure_fixture": True})
    folder = study.endpoint_folder(base, "A", 25101, 512)
    folder.mkdir(parents=True)
    (folder / "model.json").write_text("fixture model configuration", encoding="utf-8")
    (folder / "training.pt").write_bytes(b"fixture optimizer state, never deserialized")
    meta = {"mode": "A", "step": 512,
        "identity": {"mode": "A", "seed": 25101, "source_sha256": study.SOURCE_SHA,
                     "configuration": "pure endpoint fixture"},
        "model_sha256": study.sha(folder / "model.json"),
        "training_sha256": study.sha(folder / "training.pt")}
    study.write(folder / "checkpoint.json", meta)
    pointer = {"directory": folder.name, "checkpoint_sha256": study.sha(folder / "checkpoint.json")}
    study.write(folder.parent / "latest.json", pointer)
    monkeypatch.setattr(study, "checked_preparation", lambda *_, **__: ({"pure_fixture": True}, []))
    monkeypatch.setattr(study, "load_export", lambda *_: object())
    output = base / "evaluations/A_25101_step000512"
    return folder, output, pointer


def authorize_fixture_resume(base, pointer):
    study.write(base / "models/A_25101/resume_once.json", {"source": pointer,
        "maximum_replayed_unpersisted_calls": 32}, exclusive=True)


def finish_fixture_evaluation(_model, _events, output, _prepared):
    output.mkdir(parents=True, exist_ok=False)
    evidence = {"event_count": 0, "numerical_failed": True,
        "profiles": {"common385": {"all_8s_rows_qualified": False,
                                    "all_actual15s_rows_qualified": False}}}
    study.write(output / "evidence.json", evidence)
    study.write(output / "complete.json", {"status": "COMPLETED", "event_count": 0,
        "evidence_sha256": study.sha(output / "evidence.json")})
    return evidence


def test_partial_endpoint_requires_explicit_resume_and_preserves_every_original_byte(tmp_path, monkeypatch):
    _folder, output, pointer = endpoint_fixture(tmp_path, monkeypatch)
    calls = []

    def interrupted(_model, _events, directory, _prepared):
        calls.append("interrupted")
        directory.mkdir()
        np.savez(directory / "partial_arrays.npz", adverse=np.array([np.nan, 3.]))
        study.write(directory / "partial_error.json", {"original": "external interruption"})
        raise InterruptedError("simulated external interruption")

    monkeypatch.setattr(study, "evaluate_model", interrupted)
    with pytest.raises(InterruptedError):
        study.evaluate_endpoint(tmp_path, "A", 25101, 512)
    original = {str(path.relative_to(output)): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    with pytest.raises(FileExistsError, match="root-authorized"):
        study.evaluate_endpoint(tmp_path, "A", 25101, 512)
    assert not output.with_name(output.name + "_interrupted1").exists()
    authorize_fixture_resume(tmp_path, pointer)
    monkeypatch.setattr(study, "evaluate_model", finish_fixture_evaluation)
    result = study.evaluate_endpoint(tmp_path, "A", 25101, 512, resume_partial=True)
    preserved = output.with_name(output.name + "_interrupted1")
    assert {str(path.relative_to(preserved)): path.read_bytes() for path in preserved.rglob("*") if path.is_file()} == original
    assert result["numerical_failed"]
    assert (tmp_path / "models/A_25101/evaluation_resume_once.json").exists()
    assert (output / "complete.json").exists()
    assert calls == ["interrupted"]


@pytest.mark.parametrize("tamper", ("frozen_configuration", "training_state"))
def test_partial_endpoint_refuses_wrong_frozen_source_without_moving_it(tmp_path, monkeypatch, tamper):
    folder, output, pointer = endpoint_fixture(tmp_path, monkeypatch)

    def interrupted(_model, _events, directory, _prepared):
        directory.mkdir(); (directory / "partial.bin").write_bytes(b"adverse bytes")
        raise InterruptedError("external interruption")

    monkeypatch.setattr(study, "evaluate_model", interrupted)
    with pytest.raises(InterruptedError):
        study.evaluate_endpoint(tmp_path, "A", 25101, 512)
    authorize_fixture_resume(tmp_path, pointer)
    if tamper == "frozen_configuration":
        freeze = study.read(output / "frozen.json")
        freeze["configuration_sha256"] = "wrong configuration"
        study.write(output / "frozen.json", freeze)
    else:
        (folder / "training.pt").write_bytes(b"wrong training state")
    with pytest.raises(ValueError):
        study.evaluate_endpoint(tmp_path, "A", 25101, 512, resume_partial=True)
    assert (output / "science/partial.bin").read_bytes() == b"adverse bytes"
    assert not output.with_name(output.name + "_interrupted1").exists()
    assert not (tmp_path / "models/A_25101/evaluation_resume_once.json").exists()


def test_completed_numerical_failure_is_reused_and_never_reevaluated(tmp_path, monkeypatch):
    _folder, output, _pointer = endpoint_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(study, "evaluate_model", finish_fixture_evaluation)
    first = study.evaluate_endpoint(tmp_path, "A", 25101, 512)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a complete numerical failure must not be reevaluated")

    monkeypatch.setattr(study, "evaluate_model", forbidden)
    monkeypatch.setattr(study, "load_export", forbidden)
    assert study.evaluate_endpoint(tmp_path, "A", 25101, 512, resume_partial=True) == first
    assert first["numerical_failed"]
    assert not output.with_name(output.name + "_interrupted1").exists()


def test_science_completion_window_is_reconciled_without_new_predictions(tmp_path, monkeypatch):
    _folder, output, pointer = endpoint_fixture(tmp_path, monkeypatch)

    def inner_complete(_model, _events, directory, _prepared):
        finish_fixture_evaluation(_model, _events, directory, _prepared)
        raise InterruptedError("interrupted after inner complete, before outer marker")

    monkeypatch.setattr(study, "evaluate_model", inner_complete)
    with pytest.raises(InterruptedError):
        study.evaluate_endpoint(tmp_path, "A", 25101, 512)
    authorize_fixture_resume(tmp_path, pointer)
    monkeypatch.setattr(study, "load_export", lambda *_: pytest.fail("completed science must not load a new model"))
    result = study.evaluate_endpoint(tmp_path, "A", 25101, 512, resume_partial=True)
    assert result["numerical_failed"]
    assert study.read(output / "complete.json")["reconciled_science_completion_without_reevaluation"]
    assert not output.with_name(output.name + "_interrupted1").exists()


def test_freeze_is_written_before_atomic_evaluation_directory_publication(tmp_path, monkeypatch):
    _folder, output, _pointer = endpoint_fixture(tmp_path, monkeypatch)
    original_write = study.write
    frozen_writes = []

    def observed_write(path, value, **kwargs):
        path = Path(path)
        if path.name == "frozen.json":
            assert path.parent.name.startswith(".pending_")
            assert not output.exists()
            frozen_writes.append(path)
        return original_write(path, value, **kwargs)

    monkeypatch.setattr(study, "write", observed_write)
    monkeypatch.setattr(study, "evaluate_model", finish_fixture_evaluation)
    study.evaluate_endpoint(tmp_path, "A", 25101, 512)
    assert len(frozen_writes) == 1
    assert (output / "frozen.json").exists()


def test_nan_fault_snapshot_preserves_parameters_gradients_moments_and_never_latest(tmp_path):
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.value = torch.nn.Parameter(torch.tensor([float("nan")], dtype=torch.float64))
    class FakeAdam:
        def state_dict(self):
            return {"state": {0: {"step": torch.tensor(3.), "exp_avg": torch.tensor([float("nan")])}},
                    "param_groups": [{"params": [0]}]}
    model = Toy(); model.value.grad = torch.tensor([float("nan")], dtype=torch.float64)
    (tmp_path / "latest.json").write_bytes(b"unchanged safe pointer")
    record = study.quarantine_failure(tmp_path, model, FakeAdam(), np.random.default_rng(7),
        {"cursor": 2}, 3, [{"step": 2}], FloatingPointError("original nonfinite state"))
    assert record["status"] == "CAPTURED" and record["payload_saved"]
    assert not record["resumable"]
    payload_path = Path(record["directory"]) / "fault_state.pt"
    assert record["payload_sha256"] == study.sha(payload_path)
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    assert torch.isnan(payload["model_state"]["value"]).all()
    assert torch.isnan(payload["gradients"]["value"]).all()
    assert torch.isnan(payload["Adam"]["state"][0]["exp_avg"]).all()
    assert payload["step"] == 3 and payload["trace"] == [{"step": 2}]
    assert payload["numpy_generator_state"] is not None
    assert (tmp_path / "latest.json").read_bytes() == b"unchanged safe pointer"


def test_fault_capture_failure_keeps_original_error_and_does_not_claim_snapshot(tmp_path, monkeypatch):
    model = torch.nn.Linear(1, 1, dtype=torch.float64)
    class FakeAdam:
        def state_dict(self):
            return {"state": {}, "param_groups": []}
    def failed_save(*_args, **_kwargs):
        raise OSError("fixture disk capture failed")
    monkeypatch.setattr(study.torch, "save", failed_save)
    record = study.quarantine_failure(tmp_path, model, FakeAdam(), None, None, 0, [],
        FloatingPointError("original NaN failure"))
    assert not record["payload_saved"] and record["payload_sha256"] is None
    assert "original NaN failure" in record["original_error"]
    assert "fixture disk capture failed" in record["capture_error"]
    assert study.read(Path(record["directory"]) / "record.json")["status"] == "CAPTURE_FAILED"


def test_cpu_cuda_qualification_has_no_cpu_fallback(monkeypatch):
    monkeypatch.setattr(study, "resolve_device", lambda _: torch.device("cpu"))
    with pytest.raises(RuntimeError, match="requires an available CUDA"):
        study.cpu_cuda_comparison("A", [], "cpu")


def test_preflight_ineligible_modes_consume_zero_calibration_updates(tmp_path, monkeypatch):
    fit, _held = study.partition_events(events())
    panel = study.panel_events(fit)
    prepared = {"calibration_panel": [event["id"] for event in panel]}
    study.write(tmp_path / "prepare.json", {"pure_fixture": True})
    study.write(tmp_path / "preflight/summary.json", {"eligible": {mode: False for mode in study.MODES},
        "prepare_sha256": study.sha(tmp_path / "prepare.json")})
    monkeypatch.setattr(study, "checked_preparation", lambda *_, **__: (prepared, fit))
    monkeypatch.setattr(study, "code_identity", lambda: {"pure_fixture": True})
    monkeypatch.setattr(study, "build_model", lambda *_: pytest.fail("ineligible mode must not be built"))
    result = study.calibrate(tmp_path, "cpu")
    assert result["completed_updates"] == 0
    assert result["selected"] == {mode: None for mode in study.MODES}
    assert not result["no_update_cache_parsed"] and not result["no_update_data_used"]
    assert all(row["status"] == "SKIPPED_PREFLIGHT" and row["completed_updates"] == 0
        for variants in result["scores"].values() for row in variants.values())


def test_adam_state_finite_checks_reject_nonfinite_counters_and_moments():
    finite = {"step": torch.tensor(1.), "exp_avg": torch.ones(2, dtype=torch.float64),
              "exp_avg_sq": torch.ones(2, dtype=torch.float64)}
    assert study._adam_states_finite([finite])
    assert study._adam_states_finite([])
    for key in finite:
        bad = {name: value.clone() for name, value in finite.items()}
        bad[key].fill_(float("nan"))
        assert not study._adam_states_finite([bad])


def test_adam_cpu_step_cuda_moments_finite_regression_without_optimizer_step():
    if not torch.cuda.is_available():
        pytest.skip("this regression explicitly requires CPU plus CUDA tensors")
    cpu_step = torch.tensor(1., device="cpu")
    cuda_mean = torch.ones(3, dtype=torch.float64, device="cuda")
    cuda_square = torch.ones(3, dtype=torch.float64, device="cuda")
    state = {"step": cpu_step, "exp_avg": cuda_mean, "exp_avg_sq": cuda_square}
    # Retain a direct reproduction of the original mixed-device failure.
    with pytest.raises(RuntimeError, match="same device"):
        study.all_finite(list(state.values()))
    assert study._adam_states_finite([state])
    assert cpu_step.device.type == "cpu" and cuda_mean.device.type == "cuda"
    cuda_mean[1] = float("nan")
    assert not study._adam_states_finite([state])
    cuda_mean[1] = 1.
    cpu_step.fill_(float("inf"))
    assert not study._adam_states_finite([state])
