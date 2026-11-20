import json
import math

import pytest

torch = pytest.importorskip("torch")

from experiments.osgkd_fit297_20261113 import run
from kiki.models.osgkd_fitting import IntegrationWork


def constants():
    return {
        "mass": [2.0, 3.0], "damping": [0.1, 0.15], "F": [[1.0, .1], [0.0, .9]],
        "Sp": [.4, .5], "Se": [.7, .9], "omega_scale": [.4, .5], "theta_ref": [.15, -.2],
        "edge_index": [[0], [1]], "S_m": [.4, .5],
    }


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_fixed_initialization_and_optimizer_groups():
    c0, c1 = (run.new_model(variant, constants(), "cpu") for variant in run.VARIANTS)
    for key, value in c0.state_dict().items():
        assert torch.equal(value, c1.state_dict()[key])
    assert c1.c.item() == pytest.approx(.08)
    assert c1.beta.item() == pytest.approx(.008)
    assert c0.beta.item() == 0
    assert c1.init_bounds.tolist() == [4, 4, 1, 1, 4, 4]
    assert c1.config.hidden_size == 32
    assert not c1.initial_head.bias.any()
    for coeff in (c1.mechanical_coeff, c1.amplitude_coeff, c1.memory_coeff):
        assert (coeff[0] == -1).all() and not coeff[1:].any()
    optimizer = run.new_optimizer(c1)
    assert [group["lr"] for group in optimizer.param_groups] == [.001, .0003]
    assert [run.horizon(i) for i in (1, 100, 101, 200, 201, 1500)] == [20, 20, 50, 50, 200, 200]


def test_deterministic_short_final_batch():
    a, b = run.EventBatches(19), run.EventBatches(19)
    first = [a.next() for _ in range(6)]
    assert first == [b.next() for _ in range(6)]
    assert [len(batch) for batch in first] == [8, 8, 3, 8, 8, 3]
    assert sorted(index for batch in first[:3] for index in batch) == list(range(19))
    assert a.state() == b.state()


def candidate(attempt, fit, check, passed=True):
    return {"attempt": attempt, "fit": {"J": fit}, "check": {"J": check},
            "numeric": {"passed": passed}}


def test_selection_ties_and_static_fallback():
    start = candidate(0, 10, 10)
    static = candidate(400, 5, 4)
    rows = [start, candidate(200, 1, 1), static, candidate(600, 4, 4), candidate(800, 2, 2, False)]
    assert run.select_candidate(rows) is static
    assert run.select_candidate([candidate(400, 5, math.inf)]) is None
    dynamic = candidate(600, 2, 6)
    pair = run.choose_pair({"C0": {"selected": static}, "C1": {"selected": dynamic}})
    assert pair["C1_used_static_fallback"] and pair["C1"] is static and pair["C1_dynamic"] is dynamic
    tie = candidate(600, 2, 4)
    pair = run.choose_pair({"C0": {"selected": static}, "C1": {"selected": tie}})
    assert not pair["C1_used_static_fallback"] and pair["C1"] is tie


def test_adaptation_requires_reached_eligible_checkpoints_and_finite_attempts():
    records = [{"accepted": True, "clipped": True} for _ in range(1000)]
    first = candidate(600, 5, 2)
    assert run.adaptation_reason([first, candidate(1000, 4, 2)], records) == "fit_J_improved"
    assert run.adaptation_reason([first, candidate(1000, 5, 2)], records) == "reduce_lr_for_clipping"
    assert run.adaptation_reason([first, candidate(1000, 4, 2, False)], records) is None
    assert run.adaptation_reason([first, candidate(1000, 4, 2)], records[:-1]) is None
    records[-1]["accepted"] = False
    assert run.adaptation_reason([first, candidate(1000, 4, 2)], records) is None


def test_gradient_probe_directions_are_feasible_and_restore_parameters(monkeypatch):
    model = run.new_model("C1", constants(), "cpu")
    snapshot = {key: value.clone() for key, value in model.state_dict().items()}
    calls = []

    def quadratic(rom, batch, steps, scale, *, work):
        calls.append((batch[0]["id"], steps, torch.is_grad_enabled()))
        motion = sum(parameter.square().sum() for parameter in rom.parameters())
        return torch.stack((motion, motion * 0, motion * 0)).reshape(1, 3)

    monkeypatch.setattr(run, "event_objective", quadratic)
    result = run.gradient_probe(model, [{"id": "a"}, {"id": "b"}, {"id": "c"}], None, IntegrationWork())
    assert result["passed"]
    assert len(result["checks"]) == 12
    assert len(calls) == 26
    assert all(torch.equal(value, snapshot[key]) for key, value in model.state_dict().items())
    with torch.no_grad():
        model.beta.zero_()
    calls.clear()
    assert run.gradient_probe(model, [{"id": "a"}], None, IntegrationWork())["reason"] == (
        "infeasible_fixed_edge_direction"
    )
    assert calls == []


def test_nonfinite_attempt_preserves_adam_and_parameter_state(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "ROOT", tmp_path)
    device = torch.device("cpu")
    model = run.new_model("C0", constants(), device)
    optimizer = run.new_optimizer(model)
    events = [{"id": "a"}]

    def objective(rom, batch, steps, scale, *, recompute, work, path_finite):
        assert not recompute
        path_finite.extend(torch.tensor(True) for _ in batch)
        loss = rom.c.square().sum()
        return torch.stack((loss, loss * 0, loss * 0)).reshape(1, 3)

    monkeypatch.setattr(run, "event_objective", objective)
    row = run.training_attempt(model, optimizer, events, 1, None, IntegrationWork())
    assert row["accepted"] and row["projection"]["beta_zero"] == 1
    parameter_after = model.c.detach().clone()
    state_after = optimizer.state[model.c]["step"].clone()

    def inactive_state_nonfinite(rom, batch, steps, scale, *, recompute, work, path_finite):
        output = objective(rom, batch, steps, scale, recompute=recompute, work=work, path_finite=path_finite)
        path_finite[0] = torch.tensor(False)
        return output

    monkeypatch.setattr(run, "event_objective", inactive_state_nonfinite)
    row = run.training_attempt(model, optimizer, events, 2, None, IntegrationWork())
    assert not row["accepted"] and row["rejection"] == "nonfinite_trajectory"
    assert torch.equal(model.c, parameter_after)
    assert torch.equal(optimizer.state[model.c]["step"], state_after)

    batches = run.EventBatches(19)
    batches.next()
    path = tmp_path / "candidate.pt"
    ref = run._checkpoint(path, model, optimizer, batches, 2, {"audit_ref": "synthetic"}, device)
    assert run._verified_ref(ref) == path
    snapshot = torch.load(path, weights_only=True)
    assert snapshot["attempt"] == 2
    assert snapshot["numpy_batch_rng"] == batches.state()
    assert torch.equal(snapshot["torch_rng"], torch.get_rng_state())
    assert snapshot["optimizer"]["state"]


def test_checkpoint_objective_uses_original_denominator_on_numerical_failure(monkeypatch):
    def objective(model, batch, steps, scale, *, work, path_finite):
        path_finite.extend((torch.tensor(True), torch.tensor(False)))
        return torch.tensor([[1., 2., 3.], [2., 3., 4.]])

    monkeypatch.setattr(run, "event_objective", objective)
    result = run.check_objective(None, [{"id": "good"}, {"id": "bad"}], None, IntegrationWork())
    assert math.isinf(result["J"])
    assert result["eligible"] == 2 and result["completed"] == 1
    assert result["failed_ids"] == ["bad"]
    assert result["finite_subset_components"] == {"motion": 1., "ports": 2., "consistency": 3.}


def test_release_hash_binding_and_audit_gate(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(run, "REQUIRED_SOURCES", {"protocol.md"})
    source = tmp_path / "protocol.md"
    source.write_text("frozen", encoding="utf-8")
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"status": "passed"}), encoding="utf-8")
    release = {
        "run_id": run.RUN, "phase": "qualify", "job_id": "qual01", "released": True, "device": "cpu",
        "threads": 1, "torch_version": str(torch.__version__), "audit_ref": run._identity(audit),
        "source_hashes": {"protocol.md": run._identity(source)["sha256"]},
    }
    assert run.validate_release(release, "qualify", "qual01", None) == ({"status": "passed"}, {})
    with pytest.raises(ValueError, match="outside"):
        run.validate_release(release, "qualify", "another", None)
    audit.write_text(json.dumps({"status": "blocked"}), encoding="utf-8")
    release["audit_ref"] = run._identity(audit)
    with pytest.raises(ValueError, match="did not pass"):
        run.validate_release(release, "qualify", "qual01", None)
    source.write_text("mutated", encoding="utf-8")
    with pytest.raises(ValueError, match="source changed"):
        run.validate_release(release, "qualify", "qual01", None)
