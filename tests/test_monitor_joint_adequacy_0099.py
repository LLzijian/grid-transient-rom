import copy
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import monitor_joint_adequacy_0099 as monitor
from train_local_memory_0098 import objective


class SyntheticROM(torch.nn.Module):
    dim = 32

    def __init__(self):
        super().__init__()
        self.register_buffer("mass", torch.arange(1, 11, dtype=torch.float64))
        self.register_buffer("p_scale", torch.linspace(.8, 1.8, 10, dtype=torch.float64))

    def encode(self, history):
        return torch.stack((history[:, -1, 10:20].mean(-1), history[:, -1, :10].mean(-1)), -1) * .1

    def field(self, state):
        d, w, p, q = state[..., :10], state[..., 10:20], state[..., 20:30], state[..., 30:]
        return torch.cat((w, -.05*w+.02*p, -.1*p, -.2*q), -1)

    def rk4(self, state, dt):
        a = self.field(state)
        b = self.field(state+dt*a/2)
        c = self.field(state+dt*b/2)
        d = self.field(state+dt*c)
        return state+dt*(a+2*b+2*c+d)/6

    def complexity(self, state):
        return state.new_tensor(.123)


def events():
    result = []
    for i, length in enumerate((27, 36, 121, 48)):
        x = .015*np.sin(np.arange(length*30).reshape(length, 30)/71.+i)
        result.append({"id": f"e{i}", "group": str(i//2), "label": "stable" if i % 2 == 0 else "unstable",
                       "x": x, "torque": .1*x[:, 20:30]})
    return result


@pytest.fixture(autouse=True)
def no_real_sources(monkeypatch):
    monkeypatch.setattr(monitor, "sources", lambda: {})
    monkeypatch.setattr(monitor, "process_memory_bytes", lambda: 0)
    torch.set_num_threads(1)


def grid(model, data, manifest, dt):
    with torch.no_grad():
        return monitor.compute_grid(model, data, manifest, dt, .7, time.perf_counter()+60, {})


def test_manifest_fixed_seed_covers_every_event_and_leaves_global_rng():
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    data = events()
    manifest = monitor.build_manifest(data, strict=False)
    assert manifest == monitor.build_manifest(data, strict=False)
    assert sorted(r["event_index"] for b in manifest["batches"] for r in b[-2:]) == [0, 1, 2, 3]
    assert torch.equal(torch_before, torch.get_rng_state())
    assert all(np.array_equal(a, b) for a, b in zip(np_before, np.random.get_state()))
    monitor.validate_manifest(data, manifest)


def test_optimized_grid_matches_every_direct_original_objective_component():
    data, model = events(), SyntheticROM()
    manifest = monitor.build_manifest(data, strict=False)
    optimized = grid(model, data, manifest, .01)
    direct = []
    with torch.no_grad():
        for bi, batch in enumerate(manifest["batches"]):
            samples = [(data[r["event_index"]], r["endpoint"]) for r in batch]
            history = torch.as_tensor(np.stack([e["x"][end-10:end+1] for e, end in samples]))
            values = []
            for hi, h in enumerate((1., 2., 5.)):
                total, terms = objective(model, samples, history, h, .7)
                assert optimized["totals"][bi, hi] == pytest.approx(float(total), rel=1e-13, abs=1e-14)
                np.testing.assert_allclose(optimized["terms"][bi, hi], [float(terms[k]) for k in monitor.TERMS], rtol=1e-13, atol=1e-14)
                values.append(float(total))
            direct.append(np.mean(values))
    assert optimized["J"] == pytest.approx(np.mean(direct), rel=1e-13, abs=1e-14)


def test_fine_latent_step_is_ten_ms_and_local_is_point_two_seconds():
    data, model = events(), SyntheticROM()
    manifest = monitor.build_manifest(data, strict=False)
    fine = grid(model, data, manifest, .005)
    assert fine["local"]["path"].shape[1] == 21
    batch = manifest["batches"][0]
    history = torch.as_tensor(np.stack([data[r["event_index"]]["x"][r["endpoint"]-10:r["endpoint"]+1] for r in batch]))
    state = torch.cat((history[:, -1], model.encode(history)), -1)
    expected = model.rk4(model.rk4(state, .005), .005)
    np.testing.assert_allclose(fine["local"]["path"][:12, 1], expected.numpy(), atol=1e-15)
    assert fine["autonomous"]["available"].tolist() == [16, 25, 110, 37]


def test_physical_p_and_hidden_substep_failures_gate_complete_J():
    data, model = events(), SyntheticROM()
    manifest = monitor.build_manifest(data, strict=False)
    fine = grid(model, data, manifest, .005)
    coarse = copy.deepcopy(fine)
    coarse["autonomous"]["path"][0, :17, 20:30] += .1
    qa = monitor.compare_grids(model, data, fine, coarse)
    assert not qa["pass"] and qa["autonomous"][0]["p"] > .001
    coarse = copy.deepcopy(fine)
    coarse["local"]["finite"][0, :] = False
    qa = monitor.compare_grids(model, data, fine, coarse)
    assert not qa["pass"] and not qa["local"][0]["all_substeps_finite"]


def test_per_batch_tail_is_not_replaced_by_full_pool_tail():
    values = np.array([1., 2., 3., 100.])
    repeated_batch = values[[0]*10+[1, 3]]
    assert monitor._risk(repeated_batch) != monitor._risk(values)


def test_run_writes_arrays_and_valid_S_and_does_not_consume_rng(tmp_path):
    data, model = events(), SyntheticROM()
    manifest = monitor.build_manifest(data, strict=False)
    result = monitor.run_monitor(model, data, manifest, tmp_path/"run", time.perf_counter()+60,
                                 force_scale=.7)
    assert result["complete"] and result["numerical_pass"]
    assert result["J_fine"] == result["raw_J_fine"]
    assert result["rng_unchanged"] and result["sources_unchanged"]
    assert len(result["event_rows_fine"]) == 12
    assert all(v is not None for v in result["S_fine"].values())
    assert (tmp_path/"run"/"monitor_arrays.npz").exists()


def test_deadline_preserves_partial_evidence_with_null_J(tmp_path):
    data = events()
    manifest = monitor.build_manifest(data, strict=False)
    result = monitor.run_monitor(SyntheticROM(), data, manifest, tmp_path/"expired",
                                 time.perf_counter()-1, force_scale=.7)
    assert result["status"] == "interrupted"
    assert result["J_fine"] is None and not result["complete"]
    assert (tmp_path/"expired"/"evidence.json").exists()


def test_component_mismatch_cannot_be_hidden_by_exact_physical_paths():
    data, model = events(), SyntheticROM()
    manifest = monitor.build_manifest(data, strict=False)
    fine = grid(model, data, manifest, .005)
    other = copy.deepcopy(fine)
    other["terms"][0, 0, monitor.TERMS.index("latent_consistency")] += .1
    qa = monitor.compare_grids(model, data, fine, other)
    assert not qa["pass"]
    assert all(r["pass"] for r in qa["autonomous"]+qa["local"])


def test_tampered_manifest_rejected_without_resampling():
    data = events()
    manifest = monitor.build_manifest(data, strict=False)
    manifest["batches"][0][0]["endpoint"] += 1
    with pytest.raises(ValueError, match="hash"):
        monitor.validate_manifest(data, manifest)


def test_single_refinement_keeps_five_ms_score(tmp_path, monkeypatch):
    original_compare = monitor.compare_grids
    calls = []
    def controlled(model, data, fine, reference):
        calls.append(reference["dt"])
        qa = original_compare(model, data, fine, reference)
        if len(calls) == 1:
            qa["pass"] = False
        return qa
    monkeypatch.setattr(monitor, "compare_grids", controlled)
    data = events()
    result = monitor.run_monitor(SyntheticROM(), data, monitor.build_manifest(data, strict=False),
                                 tmp_path/"refine", time.perf_counter()+60, force_scale=.7)
    assert calls == [.01, .0025]
    assert result["refined"] and result["complete"] and result["numerical_pass"]
    assert result["J_fine"] == result["raw_J_fine"]
    assert result["selection_grid"] == .005


def test_refinement_deadline_retains_raw_fine_severe_diagnostics(tmp_path, monkeypatch):
    original_compute = monitor.compute_grid
    original_compare = monitor.compare_grids
    def interrupt_finer(model, data, manifest, dt, *args):
        if dt == .0025:
            raise monitor.MonitorInterrupted("synthetic finer deadline")
        return original_compute(model, data, manifest, dt, *args)
    def force_refine(*args):
        qa = original_compare(*args)
        qa["pass"] = False
        return qa
    monkeypatch.setattr(monitor, "compute_grid", interrupt_finer)
    monkeypatch.setattr(monitor, "compare_grids", force_refine)
    data = events()
    result = monitor.run_monitor(SyntheticROM(), data, monitor.build_manifest(data, strict=False),
                                 tmp_path/"partial", time.perf_counter()+60, force_scale=.7)
    assert result["status"] == "interrupted" and result["J_fine"] is None
    assert result["raw_J_fine"] is not None
    assert all(np.isfinite(value) for value in result["raw_S_fine"].values())
    assert all(value is None for value in result["S_fine"].values())
