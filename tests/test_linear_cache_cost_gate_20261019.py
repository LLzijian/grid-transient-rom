"""Bounded new-entry QA for the accepted linear-cache cost gate driver.

The driver imports this module and calls run_qa(output). All model/data used here
are synthetic; this QA never invokes a real train reader or a closed gate.
"""
from __future__ import annotations

import copy
import ctypes
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src")]
import linear_cache_cost_gate_20261019 as driver


def _expect_value_error(function, label):
    try:
        function()
    except (ValueError, TypeError):
        return
    raise AssertionError(f"expected ValueError: {label}")


def _identity_tests():
    from closure_checkpoint_identity_20261019 import identity_snapshot, identity_fingerprint
    nested = {"parameters": [("layer", (2, 3), True)], "settings": {"x": [1, 1.0, None]}}
    snapshot = identity_snapshot(nested)
    nested["parameters"][0] = ("mutated", (2, 3), True)
    assert snapshot["parameters"] == [["layer", [2, 3], True]]
    assert identity_fingerprint({"v": True}) != identity_fingerprint({"v": 1})
    assert identity_fingerprint({"v": 1}) != identity_fingerprint({"v": 1.0})
    assert identity_fingerprint({"v": [1, 2]}) != identity_fingerprint({"v": [2, 1]})
    cycle = []
    cycle.append(cycle)
    for invalid in ({1: "non-string key"}, {"n": math.inf}, {"n": float("nan")},
                    {"n": np.int64(1)}, {"n": object()}, cycle):
        _expect_value_error(lambda invalid=invalid: identity_snapshot(invalid), "identity type guard")


def _fixture_model(tmp_path, terms):
    from closure_training_runtime import TrainingClosureROM
    from closure_discovery_benchmarks import toy_source
    tmp_path.mkdir(parents=True, exist_ok=True)
    from train_closure import make_optimizer
    model = toy_source().with_terms(terms)
    source_path = tmp_path / "synthetic_source.json"
    model.export(source_path)
    trained = TrainingClosureROM.from_export(source_path)
    optimizer = make_optimizer(trained)
    rng = np.random.default_rng(7441)
    events = []
    samples = []
    for index, length in enumerate((26, 29, 33)):
        events.append({"id": f"qa-synthetic-{index}", "group": f"q{index}",
            "label": "synthetic", "x": rng.normal(0, .01, (length, 30)),
            "torque": rng.normal(0, .01, (length, 10))})
    samples = [(events[0], 10), (events[1], 11), (events[2], 10)]
    return trained, optimizer, events, samples, source_path


def _state_equal(a, b, path=""):
    if torch.is_tensor(a):
        assert torch.is_tensor(b) and a.shape == b.shape and a.dtype == b.dtype, path
        assert torch.isfinite(a).all() and torch.isfinite(b).all(), path
        if path.endswith("/step") or not a.is_floating_point():
            assert torch.equal(a.to(b.device), b), path
        else:
            torch.testing.assert_close(a.to(b.device), b, atol=2e-9, rtol=2e-9)
    elif isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), path
        for key in a:
            _state_equal(a[key], b[key], path + "/" + str(key))
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _state_equal(x, y, path + "/" + str(i))
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b), path
    elif isinstance(a, float):
        assert math.isfinite(a) and math.isfinite(b)
        assert math.isclose(a, b, rel_tol=2e-9, abs_tol=2e-9), path
    else:
        assert a == b, path


def _checkpoint_cache_test(tmp_path):
    from closure_checkpoint_identity_20261019 import (
        identity_snapshot, identity_fingerprint, save_identity_checkpoint,
        restore_identity_checkpoint,
    )
    from closure_training_runtime import TensorEventStore, TrainingClosureROM
    from linear_cache_runtime_20261019 import enable_linear_cache
    from symbolic_closure_rom import _BASE_TERMS, term_key
    from train_closure import make_optimizer, save_checkpoint, restore_checkpoint

    assert torch.cuda.is_available(), "registered cost-gate QA requires CUDA"
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    terms8 = [_BASE_TERMS[i] for i in (0, 1, 2, 3, 4, 5, 7, 8)]
    terms7 = [term for term in terms8 if term_key(term) != term_key(_BASE_TERMS[2])]
    # Exercise both allowed identities with a compact, fully synthetic fixture.
    summaries = {}
    for label, terms in (("full8", terms8), ("delete7", terms7)):
        model, _, events, samples, source_path = _fixture_model(tmp_path / label, terms)
        model = model.to(device)
        store = TensorEventStore(events, device)
        opt = make_optimizer(model)
        parameter_ids = tuple(id(p) for p in model.parameters())
        optimizer_refs = tuple(tuple(id(p) for p in group["params"]) for group in opt.param_groups)
        enabled = enable_linear_cache(model)
        assert enabled is model and tuple(id(p) for p in model.parameters()) == parameter_ids
        assert optimizer_refs == tuple(tuple(id(p) for p in group["params"]) for group in opt.param_groups)

        numpy_rng = np.random.default_rng(7442)
        coverage = {"order": [2, 0, 1], "cursor": 1, "nested": ("synthetic", [1, 2])}
        identity = identity_snapshot({"source_sha256": driver.sha(source_path),
            "code": {"synthetic_engine": "qa-fixture"},
            "data": {"ids": tuple(e["id"] for e in events), "lengths": [len(e["x"]) for e in events]},
            "structure": label, "config": {"horizon": .03, "tuple": (1, 2), "typed_flag": True},
            "parameters": [(n, tuple(p.shape), p.requires_grad) for n, p in model.named_parameters()]})
        first = driver.one_update(model, opt, store, samples, .03, .01)
        assert math.isfinite(first["loss"])
        ckpt = tmp_path / f"{label}_checkpoint"
        ckpt.mkdir(parents=True)
        torch_cpu = torch.get_rng_state().clone()
        torch_cuda = torch.cuda.get_rng_state(device).clone()
        np_state = copy.deepcopy(numpy_rng.bit_generator.state)
        model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        opt_state = copy.deepcopy(opt.state_dict())
        save_identity_checkpoint(save_checkpoint, ckpt, model, opt, numpy_rng,
                                 coverage, 1, identity, [])
        expected_refs = tuple(tuple(id(p) for p in group["params"]) for group in opt.param_groups)
        restored = driver.restore_cached(ckpt, device, identity)
        rm, ro, rr, rc, meta, changed = restored
        assert not changed and meta["step"] == 1
        assert rm is restored[0]
        _state_equal(model_state, rm.state_dict())
        _state_equal(opt_state, ro.state_dict())
        _state_equal(coverage, rc)
        _state_equal(np_state, rr.bit_generator.state)
        assert torch.equal(torch_cpu, torch.get_rng_state())
        assert torch.equal(torch_cuda, torch.cuda.get_rng_state(device))
        trainable_parameter_ids = {id(p) for p in rm.parameters() if p.requires_grad}
        restored_optimizer_ids = [id(p) for group in ro.param_groups for p in group["params"]]
        assert len(restored_optimizer_ids) == len(trainable_parameter_ids)
        assert len(set(restored_optimizer_ids)) == len(restored_optimizer_ids)
        assert set(restored_optimizer_ids) == trainable_parameter_ids
        original_names = {id(p): name for name, p in model.named_parameters()}
        restored_names = {id(p): name for name, p in rm.named_parameters()}
        original_layout = [
            (group.get("block"), group.get("lr"),
             [original_names[id(p)] for p in group["params"]])
            for group in opt.param_groups
        ]
        restored_layout = [
            (group.get("block"), group.get("lr"),
             [restored_names[id(p)] for p in group["params"]])
            for group in ro.param_groups
        ]
        assert restored_layout == original_layout

        mutations = [
            ("source_sha256", lambda value: "different-source"),
            ("code", lambda value: {**value, "synthetic_engine": "changed"}),
            ("data", lambda value: {**value, "ids": ["different-event"]}),
            ("config", lambda value: {**value, "horizon": .05}),
            ("config", lambda value: {**value, "tuple": [1.0, 2]}),
            ("config", lambda value: {**value, "typed_flag": 1}),
            ("parameters", lambda value: list(reversed(value))),
        ]
        for section, mutate in mutations:
            altered = copy.deepcopy(identity)
            altered[section] = mutate(altered[section])
            try:
                restore_identity_checkpoint(restore_checkpoint, ckpt, device, altered)
            except ValueError as exc:
                assert "typed" in str(exc)
            else:
                raise AssertionError(f"changed nested identity was accepted: {section}")

        # Branch twice from the same immutable checkpoint. Restore reestablishes
        # the global RNG state before each next update.
        next_a = driver.restore_cached(ckpt, device, identity)
        expected = driver.one_update(next_a[0], next_a[1], store, samples, .03, .01)
        state_a = {k: v.detach().clone() for k, v in next_a[0].state_dict().items()}
        adam_a = copy.deepcopy(next_a[1].state_dict())
        rng_a_cpu = torch.get_rng_state().clone()
        rng_a_cuda = torch.cuda.get_rng_state(device).clone()
        rng_a_numpy = copy.deepcopy(next_a[2].bit_generator.state)
        coverage_a = copy.deepcopy(next_a[3])
        next_b = driver.restore_cached(ckpt, device, identity)
        actual = driver.one_update(next_b[0], next_b[1], store, samples, .03, .01)
        _state_equal(state_a, next_b[0].state_dict())
        _state_equal(adam_a, next_b[1].state_dict())
        _state_equal(rng_a_numpy, next_b[2].bit_generator.state)
        _state_equal(coverage_a, next_b[3])
        assert torch.equal(rng_a_cpu, torch.get_rng_state())
        assert torch.equal(rng_a_cuda, torch.cuda.get_rng_state(device))
        assert math.isclose(expected["loss"], actual["loss"], rel_tol=2e-9, abs_tol=2e-9)
        assert expected_refs == optimizer_refs
        summaries[label] = {"initial_update": first, "next_update": actual,
                            "adam_parameter_count": len(opt.state), "step_exact": True,
                            "cache_restore": "PASS"}
    return summaries


def _budget_and_enumeration_tests():
    assert driver.validate_budget("qa", 300) == 300.
    assert driver.validate_budget("measure", 6000) == 6000.
    for mode, seconds in (("qa", 300.1), ("measure", 6000.1), ("measure", -1),
                          ("qa", float("nan")), ("other", 1)):
        _expect_value_error(lambda mode=mode, seconds=seconds: driver.validate_budget(mode, seconds), "budget")
    steps = driver.checkpoint_steps()
    assert len(steps) == 24 and steps == sorted(set(steps))
    tasks = driver.measurement_tasks()
    assert len(tasks) == 36
    assert all(set(task) == {"structure", "horizon_s", "panel", "repeat"} for task in tasks)
    assert {task["repeat"] for task in tasks} == {0, 1}
    assert {task["structure"] for task in tasks} == {"full8", "delete7"}
    assert {task["horizon_s"] for task in tasks} == {1., 2., 5.}


def _wait_owned_test():
    with tempfile.TemporaryDirectory(prefix="linear-cache-owned-tree-") as temp_dir:
        pid_path = Path(temp_dir) / "grandchild.pid"
        child_code = (
            "import subprocess,sys,time; "
            "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)'], "
            "creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)); "
            "open(sys.argv[1],'w').write(str(p.pid)); time.sleep(20)"
        )
        proc = subprocess.Popen([sys.executable, "-c", child_code, str(pid_path)],
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        deadline = time.monotonic() + 5.
        while not pid_path.exists() and proc.poll() is None and time.monotonic() < deadline:
            time.sleep(.02)
        assert pid_path.exists() and proc.poll() is None, "owned launcher did not start its child"
        grandchild_pid = int(pid_path.read_text(encoding="utf-8"))
        cleanup = {}
        try:
            driver.wait_owned(proc, .05, cleanup)
        except subprocess.TimeoutExpired:
            pass
        else:
            raise AssertionError("wait_owned did not enforce the timeout")
        assert proc.poll() is not None
        assert "returncode" in cleanup and "failure" not in cleanup

        # Verify the descendant is gone with a zero-time Windows process wait;
        # a missing PID handle also means the process has exited.
        kernel32 = ctypes.windll.kernel32
        kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel32.WaitForSingleObject.restype = ctypes.c_uint32
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        process = kernel32.OpenProcess(0x00100000, False, grandchild_pid)  # SYNCHRONIZE
        if process:
            try:
                exited = kernel32.WaitForSingleObject(process, 0) == 0
                stop = time.monotonic() + 5.
                while not exited and time.monotonic() < stop:
                    time.sleep(.02)
                    exited = kernel32.WaitForSingleObject(process, 0) == 0
                assert exited, "owned grandchild survived tree cleanup"
            finally:
                kernel32.CloseHandle(process)
        else:
            assert kernel32.GetLastError() == 87, "could not verify owned grandchild process state"
        return {"returncode": proc.returncode, "grandchild_pid": grandchild_pid,
                "grandchild_exited": True, "cleanup": cleanup}


def _cost_formula_test():
    maxima = {"full8": {1: 1., 2: 2., 5: 5.},
              "delete7": {1: .5, 2: 1., 5: 2.5}}
    overhead = {
        "full8": {"construction": 2., "save_empty": .1, "save_filled": .2,
            "restore_empty": .5, "restore_filled": .5, "full_eval": 4., "jacobian": 1.},
        "delete7": {"construction": 1., "save_empty": .05, "save_filled": .1,
            "restore_empty": .25, "restore_filled": .25, "full_eval": 2., "jacobian": .5},
    }
    rows = [{"horizon_s": h} for h in (1., 2., 5., 2.)]
    schedules = {seed: rows for seed in (6101, 6102, 6103)}
    result = driver.compute_cost(maxima, overhead, schedules, 3.)
    # Independent arithmetic: full8=34.8/model×3, delete7=18.9/model×3;
    # one recovery=max(3+.5+50×5, 3+.25+50×2.5)=253.5.
    assert len(result["models"]) == 6
    assert math.isclose(result["raw_plan_seconds"], 414.6, abs_tol=1e-10)
    assert math.isclose(result["one_global_recovery_seconds"], 253.5, abs_tol=1e-10)
    assert math.isclose(result["estimated_with_25pct_margin_seconds"], 518.25, abs_tol=1e-10)
    assert result["within_cap_arithmetic"] is True
    _expect_value_error(lambda: driver.compute_cost(maxima, overhead, schedules, -1), "negative envelope")
    missing = copy.deepcopy(overhead); del missing["full8"]["jacobian"]
    _expect_value_error(lambda: driver.compute_cost(maxima, missing, schedules, 3), "missing cost")
    incomplete = {**schedules, 6101: []}
    _expect_value_error(lambda: driver.compute_cost(maxima, overhead, incomplete, 3), "incomplete workload")
    return {"raw": 414.6, "recovery": 253.5, "with_margin": 518.25}


def run_qa(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    driver.validate_budget("qa", 300)
    driver.load_runtime()
    result = {"status": "RUNNING", "scope": "synthetic new-entry QA only",
              "real_train_read": False, "val_read": False, "test_read": False,
              "S_O_source_read": False}
    try:
        _identity_tests()
        _budget_and_enumeration_tests()
        dependency_names = set(driver.dependency_hashes())
        assert "experiments/linear_cache_runtime_20261019.py" in dependency_names
        assert "experiments/closure_checkpoint_identity_20261019.py" in dependency_names
        assert "tests/test_linear_cache_cost_gate_20261019.py" in dependency_names
        result["cache_checkpoint"] = _checkpoint_cache_test(output / "synthetic")
        result["wait_owned"] = _wait_owned_test()
        result["cost_formula"] = _cost_formula_test()
        result["code_sha256"] = driver.dependency_hashes()
        result["status"] = "PASS"
    except BaseException as exc:
        result.update(status="FAIL", error=repr(exc), traceback=traceback.format_exc())
        (output / "qa.json").write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        raise
    (output / "qa.json").write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return result
