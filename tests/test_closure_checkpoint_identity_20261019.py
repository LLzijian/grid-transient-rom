"""One-shot, synthetic CUDA checkpoint identity contract check.

Run only under the registered root-supervised invocation:
  python tests/test_closure_checkpoint_identity_20261019.py --output PATH
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src")]
from closure_checkpoint_identity_20261019 import (
    identity_snapshot, identity_fingerprint, save_identity_checkpoint,
    restore_identity_checkpoint,
)
import train_closure as legacy
from closure_training_runtime import TensorEventStore, objective
from train_local_memory_0098 import graph


def _assert_close_state(a, b):
    assert a.keys() == b.keys()
    for key in a:
        x, y = a[key], b[key]
        if torch.is_tensor(x):
            assert torch.is_tensor(y)
            assert x.dtype == y.dtype and x.shape == y.shape
            if key == "step" or not x.is_floating_point():
                assert torch.equal(x, y)
            else:
                torch.testing.assert_close(x, y, atol=2e-9, rtol=2e-9)
        elif isinstance(x, dict):
            _assert_close_state(x, y)
        else:
            assert x == y


def _update(model, opt, store, samples):
    model.zero_grad(set_to_none=True)
    opt.zero_grad(set_to_none=True)
    with graph(model):
        loss, _ = objective(model, store, samples, .03, .02)
        loss.backward()
    grad_none = {name: (param.grad is None) for name, param in model.named_parameters()}
    opt.step()
    model.end_graph_cache()
    return grad_none


def _identity(source_path, events, model):
    names = list(model.named_parameters())
    adapter = ROOT / "experiments" / "closure_checkpoint_identity_20261019.py"
    test_file = ROOT / "tests" / "test_closure_checkpoint_identity_20261019.py"
    return {
        "source_sha256": legacy.sha(source_path),
        "code": {**legacy.code_identity(),
                 "experiments/closure_checkpoint_identity_20261019.py": legacy.sha(adapter),
                 "tests/test_closure_checkpoint_identity_20261019.py": legacy.sha(test_file)},
        "data": {"synthetic_ids": tuple(e["id"] for e in events),
                 "lengths": [len(e["x"]) for e in events]},
        "config": {"nested": ("float64", ("dt", .01)), "batch": 3},
        "parameters": [(name, tuple(param.shape), param.requires_grad) for name, param in names],
    }


def _type_contract():
    original = {"nested": (1, [True, 1, 1.0, None, "x"]), "mapping": {"b": 2, "a": 3}}
    snap = identity_snapshot(original)
    assert snap == {"nested": [1, [True, 1, 1.0, None, "x"]],
                    "mapping": {"b": 2, "a": 3}}
    original["nested"][1][1] = 8
    assert snap["nested"][1][1] == 1
    assert len({identity_fingerprint({"v": x}) for x in (True, 1, 1.0)}) == 3
    assert identity_fingerprint({"p": ["a", "b"]}) != identity_fingerprint({"p": ["b", "a"]})
    for bad in ({1: "key"}, {"v": math.nan}, {"v": math.inf}, {"v": np.int64(2)}, {"v": object()}):
        try:
            identity_snapshot(bad)
        except TypeError:
            pass
        else:
            raise AssertionError(f"accepted forbidden identity value: {bad!r}")
    cycle = []
    cycle.append(cycle)
    try:
        identity_snapshot(cycle)
    except TypeError:
        pass
    else:
        raise AssertionError("accepted cyclic identity")
    from closure_checkpoint_identity_20261019 import _checked_metadata
    import tempfile
    with tempfile.TemporaryDirectory(prefix="checkpoint-path-guard-") as directory:
        root = Path(directory)
        (root / "latest.json").write_text(json.dumps({"directory": "..", "checkpoint_sha256": "x"}))
        try:
            _checked_metadata(root)
        except ValueError as error:
            assert "immediate child" in str(error)
        else:
            raise AssertionError("accepted escaping checkpoint directory")
    base = {"source": "s", "code": {"x": "h"}, "data": ["id"], "config": {"dt": .01},
            "parameters": [("w", (2, 3), True)]}
    for key, changed in (("source", "t"), ("code", {"x": "z"}), ("data", ["other"]),
                         ("config", {"dt": .02}), ("parameters", [("b", (2, 3), True), ("a", (1,), False)])):
        altered = copy.deepcopy(base)
        altered[key] = changed
        assert identity_fingerprint(base) != identity_fingerprint(altered)


def _fixture(tmp_path):
    spec = importlib.util.spec_from_file_location("closure_training_runtime_fixture",
        ROOT / "tests" / "test_closure_training_runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fixture(tmp_path)


def run_contract(output):
    assert torch.cuda.is_available(), "registered contract requires CUDA; no fallback"
    torch.set_num_threads(1)
    torch.manual_seed(19317)
    np.random.seed(19317)
    import tempfile
    with tempfile.TemporaryDirectory(prefix="closure-checkpoint-contract-") as directory:
        tmp = Path(directory)
        _, events, samples = _fixture(tmp)
        model = legacy.TrainingClosureROM.from_export(tmp / "source.json").to("cuda", dtype=torch.float64)
        device = model.mass.device
        opt = legacy.make_optimizer(model)
        rng = np.random.default_rng(51)
        coverage = {"order": [2, 0, 1], "cursor": 1, "nested": ("synthetic", [1, 2])}
        # Populate Adam moments with one real synthetic update before checkpointing.
        store = TensorEventStore(events, "cuda")
        _update(model, opt, store, samples)
        identity_original = _identity(tmp / "source.json", events, model)
        snapshot_before = identity_snapshot(identity_original)
        (tmp / "ckpt").mkdir()
        save_identity_checkpoint(legacy.save_checkpoint, tmp / "ckpt", model, opt, rng,
                                 coverage, 1, snapshot_before, [{"step": 1}])
        identity_original["config"]["nested"] = ("mutated", ("after", 7.))
        assert snapshot_before["config"]["nested"] == ["float64", ["dt", .01]]
        # Restore expected identity from the detached snapshot used by the save.
        saved_identity = copy.deepcopy(snapshot_before)
        torch_cpu = torch.get_rng_state().clone()
        torch_cuda = torch.cuda.get_rng_state().clone()
        numpy_state = copy.deepcopy(rng.bit_generator.state)
        coverage_saved = copy.deepcopy(coverage)
        restored, restored_opt, restored_rng, restored_coverage, _, changed = restore_identity_checkpoint(
            legacy.restore_checkpoint, tmp / "ckpt", device, saved_identity)
        assert not changed
        assert torch.equal(torch_cpu, torch.get_rng_state())
        assert torch.equal(torch_cuda, torch.cuda.get_rng_state())
        assert numpy_state == restored_rng.bit_generator.state
        assert coverage_saved == restored_coverage
        expected_params = saved_identity["parameters"]
        actual_params = [(name, tuple(param.shape), param.requires_grad)
                         for name, param in restored.named_parameters()]
        assert expected_params == identity_snapshot(actual_params)
        assert list(dict(model.named_parameters())) == list(dict(restored.named_parameters()))
        for (na, pa), (nb, pb) in zip(model.named_parameters(), restored.named_parameters()):
            assert na == nb and torch.equal(pa, pb)
        _assert_close_state(opt.state_dict(), restored_opt.state_dict())
        assert len(opt.state) == len(restored_opt.state)
        for section, mutate in (
            ("source_sha256", lambda x: "changed"),
            ("code", lambda x: {**x, "extra": "changed"}),
            ("data", lambda x: {**x, "synthetic_ids": ["different"]}),
            ("config", lambda x: {**x, "batch": 3.0}),
            ("parameters", lambda x: list(reversed(x))),
            ("parameters", lambda x: [row[:2] + [int(row[2])] for row in x]),
        ):
            bad = copy.deepcopy(saved_identity)
            bad[section] = mutate(bad[section])
            try:
                restore_identity_checkpoint(legacy.restore_checkpoint, tmp / "ckpt", device, bad)
            except ValueError as error:
                assert "typed" in str(error)
            else:
                raise AssertionError(f"typed identity mismatch accepted: {section}")
        # Compare the next update from two loads of the same checkpoint.
        expected_grad_none = _update(model, opt, store, samples)
        expected_model = {k: v.detach().clone() for k, v in model.state_dict().items()}
        expected_opt = copy.deepcopy(opt.state_dict())
        replay, replay_opt, _, _, _, _ = restore_identity_checkpoint(
            legacy.restore_checkpoint, tmp / "ckpt", device, saved_identity)
        replay_store = TensorEventStore(events, "cuda")
        replay_grad_none = _update(replay, replay_opt, replay_store, samples)
        assert expected_grad_none == replay_grad_none
        for key, value in expected_model.items():
            torch.testing.assert_close(value, replay.state_dict()[key], atol=2e-9, rtol=2e-9)
        _assert_close_state(expected_opt, replay_opt.state_dict())
        return {"status": "PASS", "synthetic_events": len(events), "device": str(model.mass.device),
                "identity_sha256": identity_fingerprint(saved_identity), "adam_parameters": len(opt.state),
                "tolerance": {"atol": 2e-9, "rtol": 2e-9}, "real_data_read": False,
                "val_read": False, "test_read": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [Path(__file__), Path(__file__).parents[1] / "experiments" / "closure_checkpoint_identity_20261019.py",
             Path(__file__).parents[1] / "experiments" / "train_closure.py",
             Path(__file__).parents[1] / "tests" / "test_closure_training_runtime.py"]
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    receipt = {"contract": "closure_checkpoint_identity_20261019", "source_sha256": hashes}
    try:
        _type_contract()
        receipt["checks"] = run_contract(args.output)
    except BaseException as error:
        receipt.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        raise
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
