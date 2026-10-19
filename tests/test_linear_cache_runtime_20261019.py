"""Registered linear-cache correctness, recovery and fixed CUDA throughput audit.

Root-supervised invocations only: --mode check|benchmark --output PATH.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src")]

import numpy as np
import torch

from linear_cache_runtime_20261019 import LinearCacheRuntime, enable_linear_cache
from closure_training_runtime import TrainingClosureROM, TensorEventStore, objective, export_cpu, synchronize
from train_closure import (make_optimizer, environment, save_checkpoint,
                           restore_checkpoint, sha)
from closure_checkpoint_identity_20261019 import (identity_snapshot,
    save_identity_checkpoint, restore_identity_checkpoint, identity_fingerprint)
from train_local_memory_0098 import graph
from closure_mechanism_study import source_model
from symbolic_closure_rom import _BASE_TERMS, term_key

START = time.perf_counter()
PUBLIC_TERMS = ROOT / "results" / "closure_discovery_20261017" / "E" / "packages" / "main_2701" / "model.json"
SOURCE_SHA = "c9fe3f86cf093d1f6c4600c71ecd333200840b9be4c2b156c018e1ff09a1e238"
TERMS_SHA = "3081b3eaeb3b96dd3b938d276e5dc64c83e4b023fd988fa940a817f74489973f"
TOLERANCE = {"field": {"atol": 2e-11, "rtol": 1e-11},
             "objective": {"atol": 2e-10, "rtol": 2e-10},
             "gradient_state": {"atol": 2e-9, "rtol": 2e-9}}


def _digest_obj(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                   allow_nan=False).encode()).hexdigest()


def _terms():
    public = json.loads(PUBLIC_TERMS.read_text(encoding="utf-8"))
    terms = public["terms"]
    if _digest_obj(terms) != TERMS_SHA:
        raise ValueError("registered public MAIN8 terms hash changed")
    expected = [_BASE_TERMS[i] for i in (0, 1, 2, 3, 4, 5, 7, 8)]
    if [term_key(t) for t in terms] != [term_key(t) for t in expected]:
        raise ValueError("registered MAIN8 term order changed")
    deleted = [term for term in terms if term_key(term) != term_key(_BASE_TERMS[2])]
    if len(terms) != 8 or len(deleted) != 7:
        raise ValueError("expected registered eight-term and edge-cosine-deleted structures")
    return {"full8": terms, "delete7": deleted}


def _rng_state():
    return {"cpu": torch.get_rng_state().clone(),
            "cuda": torch.cuda.get_rng_state(torch.cuda.current_device()).clone(),
            "numpy": copy.deepcopy(np.random.get_state())}


def _rng_equal(a, b):
    return (torch.equal(a["cpu"], b["cpu"]) and torch.equal(a["cuda"], b["cuda"]) and
            a["numpy"][0] == b["numpy"][0] and np.array_equal(a["numpy"][1], b["numpy"][1]) and
            a["numpy"][2:] == b["numpy"][2:])


def _environment(device):
    return {**environment(device), "python": sys.version, "numpy": np.__version__,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "default_dtype": str(torch.get_default_dtype()),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def _numpy_state_equal(a, b):
    return a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:]


def _synthetic_events():
    rng = np.random.default_rng(84101)
    events = []
    for index in range(12):
        x = np.concatenate((rng.normal(size=(531, 10)) * .02,
                            rng.normal(size=(531, 10)) * .002,
                            rng.normal(size=(531, 10)) * .02), axis=1)
        torque = rng.normal(size=(531, 10)) * .01
        events.append({"id": f"linear-cache-synthetic-{index:02d}",
                       "group": f"synthetic-{index % 3}", "label": "synthetic",
                       "x": x, "torque": torque})
    data_hash = hashlib.sha256()
    for event in events:
        data_hash.update(event["id"].encode())
        data_hash.update(np.ascontiguousarray(event["x"], dtype=np.float64).tobytes())
        data_hash.update(np.ascontiguousarray(event["torque"], dtype=np.float64).tobytes())
    specification = {"seed": 84101, "events": 12, "points_each": 531, "dt": .01,
        "x_scales": {"delta": .02, "omega": .002, "p": .02}, "torque_scale": .01,
        "distribution": "independent standard normal multiplied by fixed component scales",
        "generator": "numpy.default_rng", "endpoint": 10, "force_scale": .01,
        "generation_order_each_event": ["delta10", "omega10", "p10", "torque10"],
        "array_dtype": "float64", "flatten_order": "C"}
    data_hash.update(json.dumps(specification, sort_keys=True, separators=(",", ":")).encode())
    return events, data_hash.hexdigest(), specification


def _model_structure(model):
    return [(name, tuple(param.shape), param.requires_grad) for name, param in model.named_parameters()]


def _assert_retained_coefficients(cpu_source, model, terms):
    source_coefficients = {term_key(term): float(value.detach())
                           for term, value in zip(cpu_source.terms, cpu_source.coefficients)}
    target_coefficients = {term_key(term): float(value.detach())
                           for term, value in zip(model.terms, model.coefficients)}
    assert list(target_coefficients) == [term_key(term) for term in terms]
    for key in target_coefficients:
        if key not in source_coefficients:
            raise AssertionError("target term has no registered source coefficient")
        assert target_coefficients[key] == source_coefficients[key], "retained coefficient changed"


def _assert_tensor(a, b, atol, rtol, exact=False):
    assert a.shape == b.shape and a.dtype == b.dtype and a.device == b.device
    if exact or not a.is_floating_point():
        assert torch.equal(a, b)
    else:
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)


def _compare_nested(a, b, atol, rtol, path=""):
    assert type(a) is type(b), f"type mismatch at {path}: {type(a)} vs {type(b)}"
    if torch.is_tensor(a):
        _assert_tensor(a, b, atol, rtol, exact=path.endswith("/step"))
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b), f"array mismatch at {path}"
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), f"mapping keys differ at {path}"
        for key in a:
            _compare_nested(a[key], b[key], atol, rtol, f"{path}/{key}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), f"sequence length differs at {path}"
        for index, (x, y) in enumerate(zip(a, b)):
            _compare_nested(x, y, atol, rtol, f"{path}/{index}")
    elif isinstance(a, float):
        assert math.isfinite(a) and math.isfinite(b)
        assert math.isclose(a, b, abs_tol=atol, rel_tol=rtol), f"float mismatch at {path}"
    else:
        assert a == b, f"value differs at {path}"


def _detach_tree(value):
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, dict):
        return {k: _detach_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_detach_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_detach_tree(v) for v in value)
    return value


def _finite_tensors(value):
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, dict):
        return [tensor for item in value.values() for tensor in _finite_tensors(item)]
    if isinstance(value, (list, tuple)):
        return [tensor for item in value for tensor in _finite_tensors(item)]
    return []


def all_finite(tensors):
    # Adam's step counters are CPU tensors even when its moments and parameters
    # are on CUDA. Never stack these predicates across devices.
    return all(bool(torch.isfinite(t).all().item()) for t in tensors if t is not None)


def _cache_empty(model):
    for name in ("_graph_cache", "_runtime_cache", "_linear_cache"):
        assert getattr(model, name, None) is None, f"cache leaked: {name}"


def _compare_grads(a, b):
    assert a.keys() == b.keys()
    for name in a:
        assert (a[name] is None) == (b[name] is None), name
        if a[name] is not None:
            torch.testing.assert_close(a[name], b[name], **TOLERANCE["gradient_state"])


def _compare_models(a, b):
    assert _model_structure(a) == _model_structure(b)
    _compare_nested(a.state_dict(), b.state_dict(), **TOLERANCE["gradient_state"])
    assert all_finite([*a.state_dict().values(), *b.state_dict().values()])


def _probe(control, candidate, state):
    """Fields, external input, energy and inherited encoder on matching states."""
    with torch.no_grad():
        for z in (state, state + 1e-7 * torch.cos(torch.arange(
                40, dtype=state.dtype, device=state.device))):
            for u in (None, torch.linspace(-.003, .004, 10,
                        device=state.device, dtype=state.dtype).expand(z.shape[0], -1)):
                a, b = control.field(z, u), candidate.field(z, u)
                assert all_finite([a, b])
                torch.testing.assert_close(a, b, **TOLERANCE["field"])
            internal = control.interface_to_state(z)
            ea, ga = control.energy_and_gradient(internal)
            eb, gb = candidate.energy_and_gradient(internal)
            assert all_finite([ea, eb, ga, gb])
            torch.testing.assert_close(ea, eb, **TOLERANCE["field"])
            torch.testing.assert_close(ga, gb, **TOLERANCE["field"])
        history = state[:, :30].unsqueeze(1).expand(-1, 11, -1)
        a, b = control.encode(history), candidate.encode(history)
        assert all_finite([a, b])
        torch.testing.assert_close(a, b, **TOLERANCE["field"])
    _cache_empty(control); _cache_empty(candidate)


def _activation(model, optimizer=None):
    ident = id(model)
    params = tuple(id(p) for p in model.parameters())
    refs = None if optimizer is None else tuple(tuple(id(p) for p in g["params"])
                                               for g in optimizer.param_groups)
    result = enable_linear_cache(model)
    assert type(result) is LinearCacheRuntime
    assert result is model and id(result) == ident
    assert params == tuple(id(p) for p in result.parameters())
    if optimizer is not None:
        assert refs == tuple(tuple(id(p) for p in g["params"]) for g in optimizer.param_groups)
        assert all(id(p) in params for g in optimizer.param_groups for p in g["params"])
    _cache_empty(result)
    return result


def _lifecycle(control, candidate, state):
    """Use throwaway copies, preserving initial state for the registered objective."""
    control, candidate = copy.deepcopy(control), copy.deepcopy(candidate)
    initial = {k: v.detach().clone() for k, v in candidate.state_dict().items()}
    for values in (torch.zeros_like(candidate.coefficients),
                   torch.linspace(-.002, .002, len(candidate.terms),
                                  device=state.device, dtype=state.dtype)):
        with torch.no_grad():
            control.coefficients.copy_(values); candidate.coefficients.copy_(values)
        _probe(control, candidate, state)
    control.load_state_dict(initial); candidate.load_state_dict(initial)
    # The specialization must reject changed expression structures, including
    # a proper subset, reordered full terms and an extra structured storage spec.
    invalid_terms = [control.terms[:-1], list(reversed(control.terms))]
    for terms in invalid_terms:
        bad = control.with_terms(terms)
        try:
            enable_linear_cache(bad)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("specialized runtime accepted unregistered terms")
    for key, value in (("arm", "G3-S"), ("rank", 9)):
        bad = copy.deepcopy(control)
        setattr(bad, key, value)
        try:
            enable_linear_cache(bad)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("specialized runtime accepted an unsupported dimension/arm")
    bad = copy.deepcopy(control)
    bad.structured.storage_specs = (object(),)
    try:
        enable_linear_cache(bad)
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError("specialized runtime accepted extra storage specs")
    for terms in invalid_terms:
        try:
            candidate.with_terms(terms)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("active specialization accepted unsupported with_terms")
    _probe(control, candidate, state)
    candidate.begin_graph_cache()
    assert getattr(candidate, "_runtime_cache", None) is not None
    cache = getattr(candidate, "_linear_cache", None)
    assert cache is not None
    derived = [t for t in _finite_tensors(cache) if t.requires_grad]
    assert derived and all(t.grad_fn is not None for t in derived), "detached parameter products"
    cache_id = id(cache)
    try:
        candidate.with_terms(candidate.terms)
    except RuntimeError:
        pass
    else:
        raise AssertionError("with_terms accepted an active graph")
    candidate.field(state).square().sum().backward()
    assert id(candidate._linear_cache) == cache_id
    candidate.end_graph_cache(); _cache_empty(candidate)
    # end, load and _apply must independently discard both kinds of cache.
    candidate.begin_graph_cache(); candidate.load_state_dict(initial); _cache_empty(candidate)
    candidate.begin_graph_cache(); candidate.to("cpu"); _cache_empty(candidate)
    candidate.to(state.device); _cache_empty(candidate)
    candidate.load_state_dict(initial)
    _probe(control, candidate, state)
    with torch.no_grad():
        before = candidate.field(state).clone()
        control.coefficients.add_(.001); candidate.coefficients.add_(.001)
        after = candidate.field(state)
        assert not torch.equal(before, after), "field ignored new coefficient values"
    _probe(control, candidate, state)
    return {"zero_nonzero_fields": "PASS", "energy_encoder_external_u": "PASS",
            "graph_autograd_links": "PASS", "end_load_apply_clear": "PASS",
            "uncached_parameter_change": "PASS"}


def _loss_payload(model, store, samples, horizon, backward=True):
    model.zero_grad(set_to_none=True)
    with graph(model):
        value, terms = objective(model, store, samples, horizon, .01)
        gradients = None
        if backward:
            value.backward()
            gradients = {name: None if p.grad is None else p.grad.detach().clone()
                         for name, p in model.named_parameters()}
    model.end_graph_cache()
    _cache_empty(model)
    if not all_finite([value, *_finite_tensors(terms), *([] if gradients is None else
                   [gradient for gradient in gradients.values() if gradient is not None])]):
        raise FloatingPointError("nonfinite check objective or gradient")
    return value.detach(), _detach_tree(terms), gradients


def _one_update(model, opt, store, samples, horizon):
    opt.zero_grad(set_to_none=True)
    with graph(model):
        loss, components = objective(model, store, samples, horizon, .01)
        loss.backward()
    grads = {name: None if p.grad is None else p.grad.detach().clone()
             for name, p in model.named_parameters()}
    params = [p for p in model.parameters() if p.requires_grad]
    if not all_finite([loss, *_finite_tensors(components), *[p.grad for p in params]]):
        raise FloatingPointError("nonfinite complete objective or gradient")
    norm = torch.nn.utils.clip_grad_norm_(params, 100.)
    if not all_finite([norm]):
        raise FloatingPointError("nonfinite check clipped norm")
    opt.step()
    model.end_graph_cache()
    _cache_empty(model)
    if not all_finite(params):
        raise FloatingPointError("nonfinite Adam update")
    if not all_finite(_finite_tensors(opt.state_dict())):
        raise FloatingPointError("nonfinite Adam state")
    return grads, float(norm.detach())


def _construct_pair(cpu_source, terms, device):
    original_state = {k: v.clone() for k, v in cpu_source.state_dict().items()}
    # Query CUDA RNG before constructing either path so CUDA's lazy initialization
    # is not mistaken for RNG consumption by one constructor.
    torch.cuda.get_rng_state(device)
    torch_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone()
    numpy_rng = copy.deepcopy(np.random.get_state())
    t0 = time.perf_counter()
    control_cpu = copy.deepcopy(cpu_source)
    control = control_cpu.to(device).with_terms(terms)
    synchronize(device)
    control_seconds = time.perf_counter() - t0
    control_rng = (torch.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone(),
                   copy.deepcopy(np.random.get_state()))
    # Rewind only between paired constructors so both start from the same RNG state.
    torch.set_rng_state(torch_rng); torch.cuda.set_rng_state(cuda_rng, device); np.random.set_state(numpy_rng)
    t0 = time.perf_counter()
    candidate_cpu = copy.deepcopy(cpu_source)
    candidate = _activation(candidate_cpu.to(device).with_terms(terms))
    synchronize(device)
    candidate_seconds = time.perf_counter() - t0
    candidate_rng = (torch.get_rng_state(), torch.cuda.get_rng_state(device), copy.deepcopy(np.random.get_state()))
    if not (torch.equal(torch_rng, control_rng[0]) and torch.equal(cuda_rng, control_rng[1]) and
            _numpy_state_equal(numpy_rng, control_rng[2]) and torch.equal(torch_rng, candidate_rng[0]) and
            torch.equal(cuda_rng, candidate_rng[1]) and _numpy_state_equal(numpy_rng, candidate_rng[2])):
        raise AssertionError("a constructor changed the registered RNG states")
    if not (torch.equal(control_rng[0], candidate_rng[0]) and torch.equal(control_rng[1], candidate_rng[1]) and
            _numpy_state_equal(control_rng[2], candidate_rng[2])):
        raise AssertionError("constructor paths consumed different RNG state")
    for key, value in original_state.items():
        assert torch.equal(value, cpu_source.state_dict()[key]), f"CPU source changed: {key}"
    return control, candidate, {"control_seconds": control_seconds,
                                "candidate_seconds": candidate_seconds,
                                "cpu_source_sha256": _state_hash(cpu_source),
                                "control_rng_exact": True, "candidate_rng_exact": True}


def _state_hash(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode()); h.update(str(tensor.dtype).encode()); h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _code_hashes():
    hashes = {**__import__("train_closure").code_identity(),
        "experiments/closure_checkpoint_identity_20261019.py": sha(ROOT / "experiments" / "closure_checkpoint_identity_20261019.py"),
        "experiments/closure_mechanism_study.py": sha(ROOT / "experiments" / "closure_mechanism_study.py"),
        "experiments/shared_closure_rom_0101.py": sha(ROOT / "experiments" / "shared_closure_rom_0101.py"),
        "experiments/linear_cache_runtime_20261019.py": sha(ROOT / "experiments" / "linear_cache_runtime_20261019.py"),
        "tests/test_linear_cache_runtime_20261019.py": sha(Path(__file__))}
    # Bind local imports recursively, including function-local imports, by
    # reading their source only. This does not import a diagnostic or read data.
    pending = [ROOT / name for name in hashes]
    visited = set()

    def local_modules(name):
        parts = name.split(".") if name else []
        for base in (ROOT / "experiments", ROOT / "src"):
            for index in range(1, len(parts) + 1):
                package = base.joinpath(*parts[:index]) / "__init__.py"
                if package.is_file():
                    yield package
            module = base.joinpath(*parts).with_suffix(".py") if parts else None
            if module is not None and module.is_file():
                yield module

    while pending:
        path = pending.pop().resolve()
        if path in visited:
            continue
        visited.add(path)
        hashes[path.relative_to(ROOT).as_posix()] = sha(path)
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    if path.is_relative_to(ROOT / "src"):
                        context = list(path.relative_to(ROOT / "src").parent.parts)
                    else:
                        context = list(path.relative_to(ROOT / "experiments").parent.parts)
                    context = context[:len(context) - node.level + 1]
                    module = ".".join(context + ([module] if module else []))
                names = [module] + [module + "." + alias.name for alias in node.names
                                    if alias.name != "*"]
            else:
                continue
            for name in names:
                pending.extend(local_modules(name))
    return dict(sorted(hashes.items()))


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _identity(model, structure, source_hash, data_hash):
    return identity_snapshot({"source_sha256": source_hash,
        "code": _code_hashes(),
        "data_sha256": data_hash, "structure": structure,
        "parameters": _model_structure(model), "config": {"dtype": "float64", "dt": .01}})


def _step_state(model, opt):
    return ({k: v.detach().clone() for k, v in model.state_dict().items()}, copy.deepcopy(opt.state_dict()))


def check(output):
    if not torch.cuda.is_available():
        raise RuntimeError("registered check requires CUDA")
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    torch.cuda.get_rng_state(device)
    torch.set_num_threads(1)
    cpu_source = source_model("cpu")
    if cpu_source.mass.dtype != torch.float64 or str(cpu_source.mass.device) != "cpu":
        raise ValueError("registered source must be CPU float64")
    source_hash = SOURCE_SHA
    if _state_hash(cpu_source) == "":
        raise AssertionError("unreachable source hash")
    structures = _terms()
    events, data_hash, data_specification = _synthetic_events()
    samples = [(event, 10) for event in events]
    store = TensorEventStore(events, device)
    # Source construction may use fork_rng and seed CPU state; establish the
    # registered constructor/check RNG only after all common preparation.
    torch.manual_seed(84102); np.random.seed(84102)
    rows = {}
    progress = {"status": "RUNNING", "mode": "check", "pid": os.getpid(),
        "argv": sys.argv, "environment": _environment(device), "structures": rows,
        "real_data_read": False, "val_read": False, "test_read": False}
    _write_json(output, progress)
    for name, terms in structures.items():
        structure_name = name
        control, candidate, construction = _construct_pair(cpu_source, terms, device)
        if _model_structure(control) != _model_structure(candidate):
            raise AssertionError(f"parameter identities differ for {name}")
        buffers_control, buffers_candidate = dict(control.named_buffers()), dict(candidate.named_buffers())
        assert buffers_control.keys() == buffers_candidate.keys()
        for key, value in buffers_control.items():
            assert torch.equal(value, buffers_candidate[key]), f"buffer differs: {key}"
        for (na, pa), (nb, pb) in zip(control.named_parameters(), candidate.named_parameters()):
            assert na == nb and pa.requires_grad == pb.requires_grad
            assert torch.equal(pa, pb), f"parameter differs: {na}"
        assert list(control.state_dict()) == list(candidate.state_dict())
        for key, value in control.state_dict().items():
            _assert_tensor(value, candidate.state_dict()[key], atol=0., rtol=0., exact=True)
        assert all(p.dtype == torch.float64 for p in control.parameters())
        assert all(p.dtype == torch.float64 for p in candidate.parameters())
        assert [term_key(t) for t in control.terms] == [term_key(t) for t in candidate.terms]
        assert control.arm == candidate.arm == "L10-S"
        assert control.dim == candidate.dim == 40
        assert not control.structured.storage_specs and not candidate.structured.storage_specs
        state = torch.linspace(-.01, .01, 40, device=device, dtype=torch.float64).repeat(2, 1)
        field_a, field_b = control.field(state), candidate.field(state)
        torch.testing.assert_close(field_a, field_b, **TOLERANCE["field"])
        energy_a, grad_a = control.energy_and_gradient(state)
        energy_b, grad_b = candidate.energy_and_gradient(state)
        torch.testing.assert_close(energy_a, energy_b, **TOLERANCE["field"])
        torch.testing.assert_close(grad_a, grad_b, **TOLERANCE["field"])
        enc_a, enc_b = control.encode(state[:, :30].unsqueeze(1).expand(-1, 11, -1)), candidate.encode(state[:, :30].unsqueeze(1).expand(-1, 11, -1))
        torch.testing.assert_close(enc_a, enc_b, **TOLERANCE["field"])
        _assert_retained_coefficients(cpu_source, control, terms)
        _assert_retained_coefficients(cpu_source, candidate, terms)
        lifecycle = _lifecycle(control, candidate, state)
        comparisons = {}
        long_gradients = None
        for horizon in (.03, 5.):
            va, ca, ga = _loss_payload(control, store, samples, horizon)
            vb, cb, gb = _loss_payload(candidate, store, samples, horizon)
            torch.testing.assert_close(va, vb, **TOLERANCE["objective"])
            _compare_nested(ca, cb, **TOLERANCE["objective"])
            assert ga.keys() == gb.keys()
            for key in ga:
                assert (ga[key] is None) == (gb[key] is None), key
                if ga[key] is not None:
                    torch.testing.assert_close(ga[key], gb[key], **TOLERANCE["gradient_state"])
            if horizon == 5.:
                long_gradients = (ga, gb)
            comparisons[str(horizon)] = {"loss": float(va), "components": {k: float(v) if torch.is_tensor(v) and v.numel() == 1 else (v.tolist() if torch.is_tensor(v) else v) for k, v in ca.items()}, "finite": True}
        # Apply the already evaluated 5 s gradients first, then continue each same
        # optimizer with a fresh short update. This is one coherent update sequence.
        opt_a, opt_b = make_optimizer(control), make_optimizer(candidate)
        for model, opt, gradients in ((control, opt_a, long_gradients[0]),
                                      (candidate, opt_b, long_gradients[1])):
            for name, param in model.named_parameters():
                param.grad = None if gradients[name] is None else gradients[name].clone()
            parameters = [p for p in model.parameters() if p.requires_grad]
            norm = torch.nn.utils.clip_grad_norm_(parameters, 100.)
            assert all_finite([norm])
            opt.step()
            model.end_graph_cache(); _cache_empty(model)
        for (_, pa), (_, pb) in zip(control.named_parameters(), candidate.named_parameters()):
            torch.testing.assert_close(pa, pb, **TOLERANCE["gradient_state"])
        _compare_nested(opt_a.state_dict(), opt_b.state_dict(), **TOLERANCE["gradient_state"])
        assert all_finite([*list(control.parameters()), *list(candidate.parameters()),
                           *_finite_tensors(opt_a.state_dict()), *_finite_tensors(opt_b.state_dict())])
        grads_a, _ = _one_update(control, opt_a, store, samples, .03)
        grads_b, _ = _one_update(candidate, opt_b, store, samples, .03)
        for key in grads_a:
            assert (grads_a[key] is None) == (grads_b[key] is None)
            if grads_a[key] is not None:
                torch.testing.assert_close(grads_a[key], grads_b[key], **TOLERANCE["gradient_state"])
        for (_, pa), (_, pb) in zip(control.named_parameters(), candidate.named_parameters()):
            torch.testing.assert_close(pa, pb, **TOLERANCE["gradient_state"])
        _compare_nested(opt_a.state_dict(), opt_b.state_dict(), **TOLERANCE["gradient_state"])
        _compare_models(control, candidate)
        _probe(control, candidate, state)
        # Persist candidate's populated optimizer/RNG state and verify exact restoration.
        identity = _identity(candidate, structure_name, source_hash, data_hash)
        cpdir = output.parent / "checkpoints" / structure_name
        cpdir.mkdir(parents=True, exist_ok=True)
        checkpoint_rng = np.random.default_rng(84102)
        saved_cpu_rng = torch.get_rng_state().clone()
        saved_cuda_rng = torch.cuda.get_rng_state(device).clone()
        saved_numpy_rng = copy.deepcopy(checkpoint_rng.bit_generator.state)
        save_identity_checkpoint(save_checkpoint, cpdir, candidate, opt_b,
            checkpoint_rng, {"indices": list(range(12)), "cursor": 0}, 2, identity, [])
        restored, restored_opt, restored_np, coverage, _, changed = restore_identity_checkpoint(
            restore_checkpoint, cpdir, device, identity)
        # The legacy helper creates its original model and optimizer. Activate
        # only afterwards, retaining the exact objects already held by Adam.
        restored = _activation(restored, restored_opt)
        assert not changed
        assert coverage == {"indices": list(range(12)), "cursor": 0}
        assert torch.equal(saved_cpu_rng, torch.get_rng_state())
        assert torch.equal(saved_cuda_rng, torch.cuda.get_rng_state(device))
        assert saved_numpy_rng == restored_np.bit_generator.state
        assert np.array_equal(checkpoint_rng.integers(0, 2**31, 16),
                              restored_np.integers(0, 2**31, 16))
        assert _model_structure(restored) == _model_structure(candidate)
        assert _state_hash(restored) == _state_hash(candidate)
        _compare_nested(opt_b.state_dict(), restored_opt.state_dict(), **TOLERANCE["gradient_state"])
        grads_candidate, _ = _one_update(candidate, opt_b, store, samples, .03)
        grads_restored, _ = _one_update(restored, restored_opt, store, samples, .03)
        for key in grads_candidate:
            assert (grads_candidate[key] is None) == (grads_restored[key] is None), key
            if grads_candidate[key] is not None:
                torch.testing.assert_close(grads_candidate[key], grads_restored[key], **TOLERANCE["gradient_state"])
        for (_, pa), (_, pb) in zip(candidate.named_parameters(), restored.named_parameters()):
            torch.testing.assert_close(pa, pb, **TOLERANCE["gradient_state"])
        _compare_nested(opt_b.state_dict(), restored_opt.state_dict(), **TOLERANCE["gradient_state"])
        _probe(candidate, restored, state)
        export_path = cpdir / "post_restore_export.json"
        export_cpu(candidate, export_path)
        readback = TrainingClosureROM.from_export(export_path).to(device)
        _probe(readback, candidate, state)
        rows[structure_name] = {"construction": construction, "short_long": comparisons,
                      "identity_sha256": identity_fingerprint(identity),
                      "adam_states": len(opt_a.state), "restore_next_short_update": "PASS",
                      "cache_lifecycle": lifecycle, "legacy_export_readback": "PASS",
                      "activation_parameter_optimizer_identity": "PASS"}
        _write_json(output, progress)
    code_hashes = _code_hashes()
    return {"status": "PASS", "mode": "check", "pid": os.getpid(), "device": str(device),
        "argv": sys.argv, "check_seed": 84102,
        "environment": _environment(device), "code_sha256": _code_hashes(),
        "source_sha256": source_hash, "data_specification": data_specification,
        "public_terms_sha256": TERMS_SHA, "synthetic_data_sha256": data_hash,
        "identity": {"source_sha256": source_hash, "public_terms_sha256": TERMS_SHA,
            "synthetic_data_sha256": data_hash, "code_sha256": code_hashes},
        "events": len(events), "points_each": 531, "structures": rows,
        "tolerance": TOLERANCE, "real_data_read": False, "val_read": False, "test_read": False}


def _run_update(model, opt, store, samples, horizon):
    opt.zero_grad(set_to_none=True)
    with graph(model):
        loss, terms = objective(model, store, samples, horizon, .01)
        loss.backward()
    params = [p for p in model.parameters() if p.requires_grad]
    if not all_finite([loss, *_finite_tensors(terms), *[p.grad for p in params]]):
        raise FloatingPointError("nonfinite benchmark loss/gradient")
    norm = torch.nn.utils.clip_grad_norm_(params, 100.)
    if not all_finite([norm]):
        raise FloatingPointError("nonfinite benchmark clipped norm")
    opt.step(); model.end_graph_cache()
    _cache_empty(model)
    if not all_finite(params):
        raise FloatingPointError("nonfinite benchmark updated parameters")
    if not all_finite(_finite_tensors(opt.state_dict())):
        raise FloatingPointError("nonfinite benchmark Adam state")


def benchmark(output, check_path):
    check_record = json.loads(check_path.read_text(encoding="utf-8"))
    if check_record.get("status") != "PASS":
        raise RuntimeError("benchmark requires the registered check.json PASS")
    if not torch.cuda.is_available():
        raise RuntimeError("registered benchmark requires CUDA")
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    torch.cuda.get_rng_state(device)
    torch.set_num_threads(1)
    if check_record.get("environment") != _environment(device):
        raise RuntimeError("benchmark environment differs from the registered check")
    if (check_record.get("mode") != "check" or check_record.get("check_seed") != 84102 or
            set(check_record.get("structures", {})) != {"full8", "delete7"} or
            check_record.get("tolerance") != TOLERANCE):
        raise RuntimeError("benchmark requires a complete registered check")
    cpu_source = source_model("cpu"); structures = _terms()
    events, data_hash, data_specification = _synthetic_events()
    expected_identity = {"source_sha256": SOURCE_SHA, "public_terms_sha256": TERMS_SHA,
        "synthetic_data_sha256": data_hash, "code_sha256": _code_hashes()}
    if (check_record.get("source_sha256") != SOURCE_SHA or
            check_record.get("public_terms_sha256") != TERMS_SHA or
            check_record.get("synthetic_data_sha256") != data_hash or
            check_record.get("code_sha256") != _code_hashes() or
            check_record.get("data_specification") != data_specification or
            check_record.get("identity") != expected_identity):
        raise RuntimeError("benchmark evidence does not match the frozen source, data, terms, and code")
    store = TensorEventStore(events, device)
    samples = [(event, 10) for event in events]
    # Keep the benchmark seed after shared source/data preparation so any local
    # model-constructor seed cannot replace the registered benchmark RNG state.
    torch.manual_seed(84103); np.random.seed(84103)
    record = {"status": "RUNNING", "mode": "benchmark", "pid": os.getpid(), "device": str(device),
        "argv": sys.argv, "benchmark_seed": 84103,
        "environment": _environment(device), "source_sha256": SOURCE_SHA,
        "public_terms_sha256": TERMS_SHA, "synthetic_data_sha256": data_hash,
        "check_json_sha256": sha(check_path), "rows": []}
    record["code_sha256"] = _code_hashes()
    record["identity"] = {"source_sha256": SOURCE_SHA, "public_terms_sha256": TERMS_SHA,
        "synthetic_data_sha256": data_hash, "code_sha256": record["code_sha256"]}
    record["data_specification"] = data_specification
    record.update(real_data_read=False, val_read=False, test_read=False)
    _write_json(output, record)
    times = {name: {h: {"control": [], "candidate": []} for h in (1., 2., 5.)} for name in structures}
    initial_states = {}
    for name, terms in structures.items():
        for horizon in (1., 2., 5.):
            for order in ("AB", "BA"):
                for path in ("control", "candidate") if order == "AB" else ("candidate", "control"):
                    # Each block/path starts with the same copied CPU source and empty Adam.
                    t0 = time.perf_counter()
                    base = copy.deepcopy(cpu_source)
                    model = (base.to(device).with_terms(terms) if path == "control"
                             else enable_linear_cache(base.to(device).with_terms(terms)))
                    synchronize(device); construct_s = time.perf_counter() - t0
                    opt = make_optimizer(model)
                    assert len(opt.state) == 0
                    initial_sha = _state_hash(model)
                    if name in initial_states:
                        assert initial_sha == initial_states[name], "benchmark common initial state changed"
                    else:
                        initial_states[name] = initial_sha
                    elapsed = []
                    row = {"structure": name, "horizon_s": horizon,
                        "block_order": order, "path": path, "construction_seconds": construct_s,
                        "update_seconds": [], "completed_updates": 0,
                        "finite": True, "discarded_updates": 0,
                        "initial_state_sha256": initial_sha, "initial_adam_states": 0}
                    record["rows"].append(row)
                    _write_json(output, record)
                    for ix in range(3):
                        synchronize(device); tick = time.perf_counter()
                        _run_update(model, opt, store, samples, horizon)
                        synchronize(device); elapsed.append(time.perf_counter() - tick)
                        row["update_seconds"] = list(elapsed)
                        row["completed_updates"] = len(elapsed)
                        row["discarded_updates"] = len(elapsed)
                        _write_json(output, record)
                    times[name][horizon][path].extend(elapsed[1:])
                    row["first_update_seconds"] = elapsed[0]
                    row["repeat_update_seconds"] = elapsed[1:]
                    checkpoint_dir = output.parent / "checkpoints" / f"{name}-h{horizon:g}-{order}-{path}"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    checkpoint_identity = _identity(model, f"{name}:{path}", SOURCE_SHA, data_hash)
                    save_identity_checkpoint(save_checkpoint, checkpoint_dir, model, opt,
                        np.random.default_rng(84103), {"block_order": order, "horizon": horizon,
                        "path": path}, 3, checkpoint_identity, [])
                    row["checkpoint_directory"] = str(checkpoint_dir)
                    _write_json(output, record)
    summary = {}
    assert sum(row["completed_updates"] for row in record["rows"]) == 72
    passed = True
    for name in structures:
        control_sum = candidate_sum = 0.
        per_horizon = {}
        for horizon in (1., 2., 5.):
            assert len(times[name][horizon]["control"]) == 4
            assert len(times[name][horizon]["candidate"]) == 4
            c = float(np.median(times[name][horizon]["control"]))
            b = float(np.median(times[name][horizon]["candidate"]))
            ratio = b / c
            passed &= ratio <= 1.05
            control_sum += c; candidate_sum += b
            per_horizon[str(horizon)] = {"control_median_s": c, "candidate_median_s": b,
                                         "ratio": ratio, "under_105_percent": ratio <= 1.05}
        ratio_sum = candidate_sum / control_sum
        passed &= ratio_sum <= .90
        summary[name] = {"by_horizon": per_horizon, "control_median_sum_s": control_sum,
                         "candidate_median_sum_s": candidate_sum, "ratio": ratio_sum,
                         "under_90_percent": ratio_sum <= .90}
    record.update(status="PASS" if passed else "FAIL", summary=summary,
        throughput_gate_passed=passed, updates_discarded=72,
        timing_note="four repeated updates per structure/time horizon; construction and first update reported separately")
    _write_json(output, record)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("check", "benchmark"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    try:
        if args.mode == "check":
            record = check(output)
        else:
            check_path = output.parent / "check.json"
            record = benchmark(output, check_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False), flush=True)
    except BaseException as error:
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = {}
        if output.exists():
            try:
                partial = json.loads(output.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                partial = {}
        failure = {"status": "FAIL", "mode": args.mode, "pid": os.getpid(),
                   "error": repr(error), "traceback": traceback.format_exc(),
                   "elapsed_process_seconds": time.perf_counter() - START,
                   "code_sha256": _code_hashes(),
                   "real_data_read": False, "val_read": False, "test_read": False}
        if partial:
            partial.update(failure)
            failure = partial
        _write_json(output, failure)
        raise


if __name__ == "__main__":
    main()

