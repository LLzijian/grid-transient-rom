import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src")]

from closure_discovery_benchmarks import toy_source
from discover_closure import loss as reference_loss
from symbolic_closure_rom import SymbolicClosureROM, _BASE_TERMS
from train_local_memory_0098 import graph
from closure_training_runtime import (
    TrainingClosureROM, TensorEventStore, objective, resolve_device,
    rollout_errors, export_cpu,
)


def fixture(tmp_path, terms=None):
    rng = np.random.default_rng(721)
    source = toy_source().with_terms(_BASE_TERMS if terms is None else terms)
    with torch.no_grad():
        source.coefficients.copy_(torch.linspace(.001, .009, len(source.terms), dtype=torch.float64))
        source.template_mask[3:7] = 1.
    path = tmp_path / "source.json"
    source.export(path)
    base = TrainingClosureROM.from_export(path)
    events = []
    for index, length in enumerate((18, 33, 41)):
        events.append({
            "id": f"synthetic-{index}", "group": f"g{index}",
            "label": "unstable" if index == 1 else "stable",
            "x": rng.normal(0, .02, (length, 30)),
            "torque": rng.normal(0, .01, (length, 10)),
        })
    samples = [(events[0], 10), (events[1], 12), (events[2], 10)]
    return base, events, samples


@pytest.mark.parametrize("horizon", [.05, .2])
def test_runtime_objective_gradients_counts_and_cache_cleanup(tmp_path, horizon):
    model, events, samples = fixture(tmp_path)
    ref = SymbolicClosureROM.from_export(tmp_path / "source.json")
    store = TensorEventStore(events)
    prepared = store.prepare(samples, horizon)
    expected_counts = [min(round(horizon / .01), len(e["x"]) - 1 - i) for e, i in samples]
    assert prepared.counts.tolist() == expected_counts
    assert int(prepared.counts[0]) == min(round(horizon / .01), 18 - 1 - 10)

    with graph(ref):
        old, old_terms = reference_loss(ref, samples, __import__(
            "train_local_memory_0098").histories_for(samples), horizon, .02)
    model.begin_graph_cache()
    new, new_terms = objective(model, store, samples, horizon, .02)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    params = [dict(model.named_parameters())[n] for n in names]
    got = torch.autograd.grad(new, params, allow_unused=True)
    ref_params = dict(ref.named_parameters())
    want = torch.autograd.grad(old, [ref_params[n] for n in names], allow_unused=True)
    assert torch.allclose(new, old, atol=2e-10, rtol=2e-10)
    assert new_terms.keys() == old_terms.keys()
    for key in new_terms:
        assert torch.allclose(new_terms[key], old_terms[key], atol=2e-10, rtol=2e-10)
    for a, b in zip(got, want):
        assert (a is None) == (b is None)
        if a is not None:
            assert torch.allclose(a, b, atol=2e-9, rtol=2e-9)
    model.end_graph_cache()
    assert model._runtime_cache is None and model._graph_cache is None


def test_rollout_export_and_independent_restore(tmp_path):
    model, events, samples = fixture(tmp_path)
    store = TensorEventStore(events)
    prep = store.prepare(samples, .2)
    values = rollout_errors(model, prep)
    assert values.shape == (3,) and torch.isfinite(values).all()
    path = tmp_path / "roundtrip.json"
    export_cpu(model, path)
    restored = SymbolicClosureROM.from_export(path)
    z = torch.randn(4, 40, dtype=torch.float64) * .03
    assert torch.allclose(model.field(z), restored.field(z), atol=1e-11, rtol=0)
    history = torch.randn(2, 11, 30, dtype=torch.float64) * .02
    assert torch.allclose(model.rollout(history, 3, .01), restored.rollout(history, 3, .01), atol=1e-11, rtol=0)


def test_generic_expression_fallback(tmp_path):
    terms = [
        {"channel": "node", "expression": {"op": "mul", "args": [
            {"op": "pow2", "arg": {"op": "var", "name": "p"}},
            {"op": "pow2", "arg": {"op": "var", "name": "q"}}]}},
        {"channel": "port", "expression": {"op": "tanh", "scale": 2.,
            "arg": {"op": "var", "name": "q"}}},
    ]
    model, _, _ = fixture(tmp_path, terms)
    ref = SymbolicClosureROM.from_export(tmp_path / "source.json")
    z = torch.randn(5, 40, dtype=torch.float64) * .03
    assert torch.allclose(model.field(z), ref.field(z), atol=1e-11, rtol=0)
    assert any(k.startswith("generic_") for k in model._term_kinds)
    for field_model in (model, ref):
        field_model.field(z).square().sum().backward()
    for (_, a), (_, b) in zip(model.named_parameters(), ref.named_parameters()):
        if a.grad is not None:
            assert torch.allclose(a.grad, b.grad, atol=1e-11, rtol=1e-10)


def test_cuda_matches_cpu_when_available(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cpu, events, samples = fixture(tmp_path)
    gpu = TrainingClosureROM.from_export(tmp_path / "source.json").cuda()
    z = torch.randn(4, 40, dtype=torch.float64) * .02
    assert torch.allclose(cpu.field(z), gpu.field(z.cuda()).cpu(), atol=2e-10, rtol=0)
    store_cpu = TensorEventStore(events, "cpu")
    store_gpu = TensorEventStore(events, "cuda")
    for horizon in (.05, .2):
        a = objective(cpu, store_cpu, samples, horizon, .02)[0]
        b = objective(gpu, store_gpu, samples, horizon, .02)[0]
        assert torch.allclose(a, b.cpu(), atol=2e-9, rtol=2e-9)
        a.backward(); b.backward()
        for (_, p), (_, q) in zip(cpu.named_parameters(), gpu.named_parameters()):
            if p.grad is not None:
                assert torch.allclose(p.grad, q.grad.cpu(), atol=2e-8, rtol=2e-8)
        cpu.zero_grad(set_to_none=True); gpu.zero_grad(set_to_none=True)
    assert resolve_device("auto").type == "cuda"


def test_device_and_state_changes_invalidate_graph_cache(tmp_path):
    model, _, _ = fixture(tmp_path)
    state = model.state_dict()
    model.begin_graph_cache()
    model.to("cpu")
    assert model._runtime_cache is None
    model.begin_graph_cache()
    model.load_state_dict(state)
    assert model._runtime_cache is None
    model.begin_graph_cache()
    with pytest.raises(RuntimeError, match="after closing"):
        export_cpu(model, tmp_path / "bad.json")
    model.end_graph_cache()


@pytest.mark.parametrize("name", ["main_2701", "fixed_2701", "alternative_2701", "delete2_2701"])
def test_published_frozen_equations_on_synthetic_states(name):
    # Public model parameters only; no private observation arrays or val/test.
    path = ROOT / "results/closure_discovery_20261017/E/packages" / name / "model.json"
    if not path.exists():
        pytest.skip("optional published model package is absent")
    ref = SymbolicClosureROM.from_export(path)
    fast = TrainingClosureROM.from_export(path)
    z = .02 * torch.sin(torch.arange(120, dtype=torch.float64).reshape(3, 40))
    with graph(ref), graph(fast):
        torch.testing.assert_close(fast.field(z), ref.field(z), rtol=1e-11, atol=2e-11)
    x = z[0].requires_grad_(True)
    a = torch.autograd.functional.jacobian(fast.field, x)
    b = torch.autograd.functional.jacobian(ref.field, x)
    torch.testing.assert_close(a, b, rtol=1e-10, atol=2e-10)
