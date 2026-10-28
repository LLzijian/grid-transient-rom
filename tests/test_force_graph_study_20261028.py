"""Data-free study checks: no real events and zero optimizer.step calls.

Fake checkpoint states test serialization mechanics; registered real16 CUDA QA
is a separate native phase and is never claimed by these checks.
"""
import copy
import importlib.util
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import force_graph_study_20261028 as study
from force_graph_model_20261028 import ForceGraphROM, MODES

torch.set_num_threads(1)


def physical_config():
    pairs = [(i, j) for i in range(10) for j in range(i + 1, 10)]
    D = np.zeros((26, 10))
    for row, (i, j) in zip(D, pairs):
        row[i], row[j] = 1., -1.
    return {"mass": np.linspace(.7, 1.9, 10).tolist(), "D": D.tolist(), "coupling": [.4] * 26,
        "delta_star": np.linspace(-.1, .1, 10).tolist(), "damping": [.2] * 10,
        "jpi_p": [1.] * 10, "rpp": [.4] * 10, "storage_base": [1.2] * 10,
        "input_scale_delta": [.2] * 10, "input_scale_omega": [.3] * 10,
        "p_scale": [.7] * 10, "gain": [1.] * 10, "rate": [.5] * 10}


@pytest.fixture(autouse=True)
def forbid_actual_optimizer_calls(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("pre-freeze tests must not invoke optimizer.step")
    monkeypatch.setattr(torch.optim.Adam, "step", forbidden)


def profile(r5=1., r8=1., *, qualified=True, perturb=.0001, missed=1, false=1, unstable=1., params=47, cost=1.):
    return {"horizon_risks": {"5": {"value": r5, "numerical_score_perturbation": perturb},
        "8": {"value": r8, "numerical_score_perturbation": perturb}},
        "classification_8s": {"missed_resolved_unstable": missed, "false_los_resolved_stable": false},
        "all_8s_rows_qualified": qualified, "unstable_interval_risk": unstable,
        "common_frequency_rmse_8s": .1, "normalized_p_rmse_8s": .2,
        "parameter_count": params, "qualified_inference_seconds_per_event": cost}


def trace(*, clips=0, jumps=0):
    return [{"gradient_norm": 101. if i < clips else 1.,
        "block_updates": {"a_bar": {"relative_l2": .251 if i < jumps else .01}}} for i in range(48)]


@pytest.mark.parametrize("step,h", [(1, 5.), (2, 8.), (191, 5.), (192, 8.)])
def test_frozen_horizon(step, h):
    assert study.horizon_for(step) == h


@pytest.mark.parametrize("step", [0, 193, True, 2.])
def test_reject_unregistered_horizon(step):
    with pytest.raises(ValueError):
        study.horizon_for(step)


def test_fixed_schedule_coverage_starts_and_ragged_counts():
    events = [{"id": f"fit{i:03d}", "x": np.zeros((12 + i % 820, 30))} for i in range(297)]
    schedules = [study.make_schedule(events, seed) for seed in study.SEEDS]
    assert study.digest(schedules[0]) == study.digest(study.make_schedule(events, 28101))
    assert study.digest(schedules[0]) != study.digest(schedules[1])
    for schedule in schedules:
        assert len(schedule["rows"]) == 192
        assert set(i for row in schedule["rows"][:75] for i in row["coverage_indices"]) == set(range(297))
        for row in schedule["rows"]:
            assert len(row["sample_ids"]) == 12 and len(row["random_indices"]) == 8
            assert len(set(row["coverage_indices"])) == 4
            assert row["endpoints"] == [10] * 12
            assert all(0 < n <= row["horizon_s"] * 100 for n in row["genuine_steps"])


def test_panel_roundrobin_group_id_and_strict_counts():
    events = [{"id": f"{label}_{g}_{i:02d}", "group": g, "label": label,
        "x": np.zeros((811 if label == "stable" else 14, 30))}
        for label in ("stable", "unstable") for g in ("a", "b", "c") for i in range(15)]
    panel = study.panel_events(list(reversed(events)))
    assert [e["group"] for e in panel[:6]] == ["a", "b", "c"] * 2
    assert sum(e["label"] == "stable" for e in panel) == 36
    assert sum(e["label"] == "unstable" for e in panel) == 12
    assert len({e["id"] for e in panel}) == 48
    with pytest.raises(ValueError, match="ESCALATE"):
        study.panel_events([e for e in events if e["group"] == "a"])


def test_objective_uses_once_only_initialization_real_intervals_and_ceil_risk(monkeypatch):
    model = ForceGraphROM(physical_config(), "M")
    events = study.old.synthetic_fixture(); store = study.EventStore(events, "cpu")
    count = [0]; initialize = model.initialize
    def once(history):
        count[0] += 1
        return initialize(history)
    monkeypatch.setattr(model, "initialize", once)
    with study.graph(model):
        losses = study.event_losses(model, store, [(e, 10) for e in events], .08)
    assert count[0] == 1
    manual = []
    for event in events:
        n = min(8, len(event["x"]) - 11)
        prediction = model.rollout(event["x"][:11], n, .01)
        manual.append(study._point_error(prediction[1:], torch.as_tensor(event["x"][11:11 + n]), model).mean())
    torch.testing.assert_close(losses, torch.stack(manual), atol=1e-12, rtol=1e-12)
    with study.graph(model):
        loss, terms = study.objective(model, store, [(e, 10) for e in events], .08)
    torch.testing.assert_close(loss, 10 * (.8 * losses.mean() + .2 * losses.max()))
    assert set(terms) == {"risk", "mean_point_error", "objective_J"}


def test_event_store_ignores_torque_and_rejects_duplicate_identity():
    events = study.old.synthetic_fixture()
    for e in events:
        e.pop("torque")
    assert len(study.EventStore(events, "cpu").events) == 2
    with pytest.raises(ValueError):
        study.EventStore([events[0], events[0]], "cpu")


def test_extension_all_registered_safety_and_perturbation_gates():
    assert study.extension_decision(profile(), profile(.98, 1.04), trace())["extend"]
    for after in (profile(.98, 1.051), profile(.98, missed=2), profile(.98, false=2),
                  profile(.98, unstable=1.051), profile(.98, qualified=False), profile(.999)):
        assert not study.extension_decision(profile(), after, trace())["extend"]
    assert not study.extension_decision(profile(perturb=.011), profile(.98, perturb=.011), trace())["extend"]
    assert study.extension_decision(profile(), profile(.98), trace(clips=12))["extend"]
    assert not study.extension_decision(profile(), profile(.98), trace(clips=13))["extend"]
    assert study.extension_decision(profile(), profile(.98), trace(jumps=11))["extend"]
    assert not study.extension_decision(profile(), profile(.98), trace(jumps=12))["extend"]


def test_each_model_calibration_independent_and_ties_prefer_one():
    scores = {"L": {"1": {"qualified": True, "ratio": .9}, "3": {"qualified": True, "ratio": .9}},
        "S": {"1": {"qualified": False, "ratio": .1}, "3": {"qualified": True, "ratio": .8}},
        "M": {"1": {"qualified": False, "ratio": .1}, "3": {"qualified": False, "ratio": .1}}}
    assert study.select_multipliers(scores) == {"L": 1, "S": 3, "M": None}


def test_backup_selection_only_qualified_gain_and_competitive_control():
    profiles = {"L": profile(params=2), "S": profile(.97, .99), "M": profile(.98, .99, cost=2.)}
    selected = study.select_backup(profiles)
    assert selected["structure"] == "S" and selected["control"] == "M"
    profiles["S"] = profile(.97, 1.06)
    assert study.select_backup(profiles)["structure"] == "M"
    profiles["M"] = profile(.98, false=2)
    assert study.select_backup(profiles)["structure"] is None
    assert study.select_backup({"S": profile(.1)})["structure"] is None


def test_backup_dominance_and_numerical_unresolved_cannot_win():
    profiles = {"L": profile(params=2), "S": profile(.97, .99), "M": profile(.96, .98)}
    assert study.select_backup(profiles)["structure"] == "M"
    profiles["M"]["all_8s_rows_qualified"] = False
    assert study.select_backup(profiles)["structure"] == "S"
    profiles["L"]["all_8s_rows_qualified"] = False
    assert study.select_backup(profiles)["structure"] is None


def test_nonnegative_projection_does_not_mutate_saved_moments():
    model = ForceGraphROM(physical_config(), "M")
    moment = torch.full_like(model.a_bar, -.3); before = moment.clone()
    with torch.no_grad():
        model.a_bar.fill_(-.2)
    model.project_nonnegative_()
    assert torch.count_nonzero(model.a_bar) == 0
    assert torch.equal(moment, before)


class FakeAdam:
    """Serialization-only fixture. No optimizer operation is implemented."""
    def __init__(self, model, multiplier=1):
        self.param_groups = [{"params": [p], "block": n, "lr": .01 * multiplier} for n, p in model.named_parameters()]
        self.state = {g["params"][0]: {"step": torch.tensor(16.),
            "exp_avg": torch.full_like(g["params"][0], .125), "exp_avg_sq": torch.full_like(g["params"][0], .0625)} for g in self.param_groups}
    def state_dict(self):
        return {"state": {i: copy.deepcopy(self.state[g["params"][0]]) for i, g in enumerate(self.param_groups)},
            "param_groups": [{**{k: v for k, v in g.items() if k != "params"}, "params": [i]} for i, g in enumerate(self.param_groups)]}
    def load_state_dict(self, state):
        for i, group in enumerate(self.param_groups):
            self.state[group["params"][0]] = copy.deepcopy(state["state"][i])


def test_checkpoint_layout_counter_rng_and_hash_checks(tmp_path, monkeypatch):
    model = ForceGraphROM(physical_config(), "S"); opt = FakeAdam(model); rng = np.random.default_rng(55)
    coverage = {"cover_order": [1, 0], "cursor": 1}; identity = {"fixture_fake_state_only": True}
    monkeypatch.setattr(study, "make_optimizer", FakeAdam)
    study._seed(55, "cpu")
    snapshot = study.old._cpu_snapshot(study._rng_state(rng, coverage))
    folder = study.save_checkpoint(tmp_path, model, opt, rng, coverage, 16, identity, [])
    random.random(); np.random.random(); torch.rand(1)
    restored, restored_opt, restored_rng, restored_coverage, meta = study.restore_checkpoint(tmp_path, "cpu", identity)
    study.old._exact_nested(snapshot, study.old._cpu_snapshot(study._rng_state(restored_rng, restored_coverage)))
    study.old._exact_nested(opt.state_dict(), restored_opt.state_dict())
    assert meta["step"] == 16
    study.old._exact_nested(model.state_dict(), restored.state_dict())
    with pytest.raises(FileExistsError):
        study.save_checkpoint(tmp_path, model, opt, rng, coverage, 16, identity, [])
    with pytest.raises(ValueError, match="identity"):
        study.restore_checkpoint(tmp_path, "cpu", {"different": True})
    (folder / "model.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        study.restore_checkpoint(tmp_path, "cpu", identity)


def test_checkpoint_adam_counter_mismatch_rejected(tmp_path, monkeypatch):
    model = ForceGraphROM(physical_config()); opt = FakeAdam(model)
    for state in opt.state.values():
        state["step"] = torch.tensor(15.)
    monkeypatch.setattr(study, "make_optimizer", FakeAdam)
    study.save_checkpoint(tmp_path, model, opt, np.random.default_rng(1), {}, 16, {}, [])
    with pytest.raises(ValueError, match="counter"):
        study.restore_checkpoint(tmp_path, "cpu", {})


def test_quarantine_preserves_nonfinite_state_without_resume_pointer(tmp_path):
    model = ForceGraphROM(physical_config()); opt = FakeAdam(model)
    with torch.no_grad():
        model.kappa_raw.fill_(float("nan"))
    result = study.old.quarantine_failure(tmp_path, model, opt, np.random.default_rng(1), {}, 0, [], ValueError("fixture failure"))
    assert result["status"] == "CAPTURED" and not result["resumable"]
    assert not (tmp_path / "latest.json").exists()
    payload = torch.load(Path(result["directory"]) / "fault_state.pt", weights_only=False)
    assert torch.isnan(payload["model_state"]["kappa_raw"])


def test_optimizer_call_ledger_counts_actual_categories_and_bounds(tmp_path):
    study._journal(tmp_path, "optimizer_step_attempt", "formal", mode="L", step=1)
    study._journal(tmp_path, "optimizer_step_completed", "formal", mode="L", step=1)
    study._journal(tmp_path, "optimizer_step_attempt", "qa", mode="M", step=1)
    assert study.call_counts(tmp_path) == {"attempts": 2, "completed": 1,
        "by_category": {"formal": {"attempts": 1, "completed": 1}, "qa": {"attempts": 1, "completed": 0}}}
    with (tmp_path / "optimizer_calls.jsonl").open("w") as f:
        for _ in range(108):
            f.write('{"event":"optimizer_step_attempt","category":"qa"}\n')
    with pytest.raises(ValueError, match="category"):
        study._journal(tmp_path, "optimizer_step_attempt", "qa")


def test_common_method_requires_fine_rk4_reference(tmp_path):
    output = tmp_path / "qualification"; output.mkdir()
    study.write(output / "summary.json", {"status": "COMPLETED", "training_method": "rk4", "all_reference_qualified": True, "all_fast_qualified": False})
    assert study._method(tmp_path) == "rk4"
    study.write(output / "summary.json", {"status": "COMPLETED", "training_method": "fast", "all_reference_qualified": True, "all_fast_qualified": False})
    with pytest.raises(ValueError):
        study._method(tmp_path)
    study.write(output / "summary.json", {"status": "UNRESOLVED", "training_method": "rk4", "all_reference_qualified": False, "all_fast_qualified": False})
    with pytest.raises(ValueError):
        study._method(tmp_path)


def test_import_is_data_and_optimizer_inert(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("import performed external work")
    monkeypatch.setattr(study.old, "checked_preparation", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    spec = importlib.util.spec_from_file_location("_inert_force_graph_study", Path(study.__file__))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    assert all(callable(getattr(module, name)) for name in ("prepare", "qualify", "qa", "calibrate", "study", "summary"))
