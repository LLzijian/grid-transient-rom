"""Synthetic QA for the fixed-edge paired training runner's new wiring."""
from __future__ import annotations

import copy
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src")]
import fixed_edge_pair_20261019 as runner


def _raises(fn, label):
    try:
        fn()
    except (ValueError, TypeError, FileExistsError, FloatingPointError):
        return
    raise AssertionError(f"expected rejection: {label}")


def _identity_budget_tests(tmp):
    assert runner.task_order() == [
        {"seed": seed, "structure": structure}
        for seed in (6101, 6102, 6103) for structure in ("full8", "delete7")]
    assert runner.model_cap("full8") == 12000 and runner.model_cap("delete7") == 10000
    assert runner.validate_budget("qa", 600) == 600
    assert runner.validate_budget("campaign", 69000) == 69000
    assert runner.validate_budget("analyze", 1800) == 1800
    for args in (("qa", 600.1), ("campaign", 69000.1), ("analyze", 1800.1),
                 ("campaign", 0), ("unknown", 1)):
        _raises(lambda args=args: runner.validate_budget(*args), "category cap")
    assert runner.lr_multiplier(512) == 1 and runner.lr_multiplier(513) == .3
    steps = runner.checkpoint_steps()
    assert len(steps) == 24 and steps == sorted(set(steps))
    assert steps == sorted({0, *range(50, 1001, 50), 256, 512, 1024})
    assert set(steps) & {256, 512, 1024} == {256, 512, 1024}
    assert runner.MILESTONES == (256, 512, 1024) and len(set(runner.MILESTONES)) == 3
    for previous, attempt, structure, replayed, recovery, used in (
        (0, 12001, "full8", 0, 0, False),
        (0, 10, "delete7", 51, 0, False),
        (0, 10, "full8", 0, 1201, False),
        (100, 200, "full8", 50, 1200, False),
        (0, 10, "full8", 0, 1, True),
    ):
        _raises(lambda: runner.recovery_accounting(previous, attempt, structure,
                replayed, recovery, used_before=used), "recovery bound")
    receipt = runner.recovery_accounting(100, 1400, "full8", 50, 1200)
    assert receipt["model_seconds"] == 1500 and receipt["recovery_seconds_subset"] == 1200
    lock = runner.claim_queue(tmp / "queue")
    assert lock.is_file()
    _raises(lambda: runner.claim_queue(tmp / "queue"), "duplicate queue")
    assert runner.pause_requested(tmp / "queue") is False
    (tmp / "queue" / "pause.request").write_text("pause at next safe update\n", encoding="utf-8")
    assert runner.pause_requested(tmp / "queue") is True
    expected = {"source": "synthetic", "config": {"n": 3, "flag": True}}
    runner.verify_identity(expected, copy.deepcopy(expected))
    altered = copy.deepcopy(expected); altered["config"]["n"] = 3.0
    _raises(lambda: runner.verify_identity(expected, altered), "typed identity")


def _risk_and_unit_tests():
    # Stable complete-5s objective-point risks: risk([1,4])=.8*2.5+.2*4=2.8.
    rows = [
        {"event_id": "a", "label": "stable", "horizon": "5", "available": True,
         "finite": True, "numerically_admissible": True, "mean_point_error": 1.,
         "records_normalized_rmse": 900.},
        {"event_id": "b", "label": "stable", "horizon": 5, "available": True,
         "finite": True, "numerically_admissible": True, "mean_point_error": 4.,
         "records_normalized_rmse": .001},
        {"event_id": "u", "label": "unstable", "horizon": "5", "available": True,
         "finite": True, "numerically_admissible": True, "mean_point_error": 100.},
        {"event_id": "short", "label": "stable", "horizon": "5", "available": False,
         "finite": None, "numerically_admissible": False, "mean_point_error": None},
    ]
    qualified = {"a": True, "b": True, "u": False}
    result = runner.primary_risk(rows, qualified)
    assert result["eligible"] == 2 and math.isclose(result["value"], 2.8, abs_tol=1e-12)
    unresolved = runner.primary_risk(rows, {"a": True, "b": False})
    assert unresolved["value"] is None and unresolved["unresolved_ids"] == ["b"]
    assert runner.primary_risk([rows[2]], {})["value"] is None

    # Qualification includes every available numeric horizon and the pair row.
    passing = {"available": True, "finite_to_endpoint": True, "los_pair_pass": True,
               "coi_normalized_rmse": .0001, "common_normalized_rmse": .0001}
    failed_scored = {**passing, "horizon": "2", "available": True, "los_pair_pass": False}
    assert runner.scientific_qualification(passing, [passing]) is True
    assert runner.scientific_qualification(passing, [{**failed_scored, "available": False}]) is True
    assert runner.scientific_qualification(passing, [failed_scored]) is False

    import numpy as np
    pred = np.zeros((126, 30), dtype=np.float64)
    event = {"id": "synthetic", "group": "g", "label": "stable",
             "x": np.zeros((126, 30), dtype=np.float64)}
    pred[1:, 20:30] = .02
    mass = np.ones(10, dtype=np.float64)
    first = runner.physical_metrics(pred, event, mass, np.ones(10))
    second = runner.physical_metrics(pred, event, mass, np.full(10, 2.))
    assert first["5"]["mechanical_power"]["unit"] == "p.u. (100 MVA system base)"
    assert math.isclose(first["1"]["mechanical_power"]["rmse_pu"], .02, abs_tol=1e-12)
    assert first["1"]["mechanical_power"]["rmse_pu"] == second["1"]["mechanical_power"]["rmse_pu"]
    assert first["1"]["mechanical_power"]["normalized_mse"] != second["1"]["mechanical_power"]["normalized_mse"]
    _raises(lambda: runner.physical_metrics(pred, event, mass, np.ones(10), mva_base=50),
            "wrong MVA base")


def _metadata_tests():
    import numpy as np
    reference = {"mva_base": 100, "equilibrium": {"delta": np.linspace(-.1, .1, 10)},
                 "inertia": np.linspace(1., 2., 10), "omega_s": 2*math.pi*60}
    event = {"id": "synthetic-unstable", "label": "unstable",
             "x": np.zeros((26, 30)), "event_time_s": .15}
    meta = {"equilibrium_delta": reference["equilibrium"]["delta"].copy(),
            "inertia": reference["inertia"].copy(), "omega_s": reference["omega_s"],
            "event_time_s": .15}
    assert runner.validate_classification_metadata(event, meta, reference) is True
    for bad_reference, bad_meta, bad_event, label in (
        ({**reference, "mva_base": 50}, meta, event, "system MVA"),
        ({**reference, "equilibrium": {"delta": reference["equilibrium"]["delta"] + .01}}, meta, event, "equilibrium"),
        ({**reference, "inertia": reference["inertia"] + .1}, meta, event, "inertia"),
        (reference, {**meta, "omega_s": 60.}, event, "frequency"),
        (reference, {**meta, "event_time_s": .30}, event, "event time range"),
    ):
        _raises(lambda ref=bad_reference, m=bad_meta, e=bad_event:
                runner.validate_classification_metadata(e, m, ref), label)
    stable = {"id": "synthetic-stable", "label": "stable", "x": np.zeros((26, 30))}
    _raises(lambda: runner.validate_classification_metadata(stable,
        {**meta, "event_time_s": .01}, reference), "stable with event time")


def _sampler_checkpoint_test(output):
    import numpy as np
    import torch
    from closure_discovery_benchmarks import toy_source
    from closure_training_runtime import TensorEventStore, TrainingClosureROM
    from symbolic_closure_rom import _BASE_TERMS
    from train_closure import make_optimizer
    from train_joint_numerical_pair_0100 import next_samples

    if not torch.cuda.is_available():
        raise RuntimeError("registered paired-runner QA requires CUDA")
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    terms = [_BASE_TERMS[i] for i in (0, 1, 2, 3, 4, 5, 7, 8)]
    output.mkdir(parents=True, exist_ok=True)
    source_path = output / "synthetic_source.json"
    toy_source().with_terms(terms).export(source_path)
    model = TrainingClosureROM.from_export(source_path).to(device)
    params = tuple(model.named_parameters())
    opt = make_optimizer(model)
    names = {id(p): n for n, p in params}
    expected_trainable = {id(p) for _, p in params if p.requires_grad}
    groups = [
        (group["block"], group["lr"], [names[id(p)] for p in group["params"]])
        for group in opt.param_groups]
    flattened = [id(p) for group in opt.param_groups for p in group["params"]]
    assert len(flattened) == len(set(flattened)) and set(flattened) == expected_trainable
    assert groups and all(group[2] for group in groups)

    rng_data = np.random.default_rng(83101)
    events = [{"id": f"qa-{i}", "group": f"g{i}", "label": "stable",
               "x": rng_data.normal(0, .01, (length, 30)),
               "torque": rng_data.normal(0, .01, (length, 10))}
              for i, length in enumerate((26, 29, 33))]
    store = TensorEventStore(events, device)
    rng = np.random.default_rng(83102)
    coverage = {"cover_order": [0, 1, 2], "cursor": 0}
    initial_rng = copy.deepcopy(rng.bit_generator.state)
    initial_coverage = copy.deepcopy(coverage)
    schedule = []
    for step in (1, 2):
        samples, sampling = next_samples(events, rng, coverage)
        schedule.append({"step": step, "horizon_s": 1. if step == 1 else 2.,
                         "sampling": sampling,
                         "numpy_rng_after": copy.deepcopy(rng.bit_generator.state),
                         "long_counts": [min(round((1. if step == 1 else 2.)*100), len(e["x"])-11) for e, _ in samples],
                         "local_counts": [min(20, len(e["x"])-i-1) for e, i in samples]})
    pair_rng = np.random.default_rng(83102)
    pair_coverage = copy.deepcopy(initial_coverage)
    paired_schedule = []
    for step in (1, 2):
        samples, sampling = next_samples(events, pair_rng, pair_coverage)
        paired_schedule.append({"step": step, "horizon_s": 1. if step == 1 else 2.,
            "sampling": sampling, "numpy_rng_after": copy.deepcopy(pair_rng.bit_generator.state),
            "long_counts": [min(round((1. if step == 1 else 2.)*100), len(e["x"])-11) for e, _ in samples],
            "local_counts": [min(20, len(e["x"])-i-1) for e, i in samples]})
    assert paired_schedule == schedule, "paired synthetic structures must share the frozen sampler schedule"
    rng.bit_generator.state = copy.deepcopy(initial_rng)
    coverage = copy.deepcopy(initial_coverage)

    identity = {"source": "synthetic-only", "code": runner.dependency_hashes(),
                "data": {"ids": [e["id"] for e in events], "lengths": [len(e["x"]) for e in events]},
                "config": {"structure": "full8", "seed": 83102},
                "parameters": [(n, tuple(p.shape), p.requires_grad) for n, p in params]}
    cpdir = output / "checkpoint"
    cpdir.mkdir(parents=True, exist_ok=True)
    trace = runner.train_steps(model, opt, store, events, rng, coverage, schedule[:1],
        end_step=0, save=lambda step, history: runner.save_safe(
            cpdir, model, opt, rng, coverage, step, identity, history))
    assert trace == []
    latest = (cpdir / "latest.json").read_bytes()
    _raises(lambda: runner.save_safe(cpdir, model, opt, rng, coverage, 1, identity, [], safe=False),
            "unsafe checkpoint")
    assert (cpdir / "latest.json").read_bytes() == latest

    rng.bit_generator.state = copy.deepcopy(initial_rng)
    coverage = copy.deepcopy(initial_coverage)
    trace_a = runner.train_steps(model, opt, store, events, rng, coverage, schedule[:1], end_step=1)
    state_a = {k: v.detach().clone() for k, v in model.state_dict().items()}
    adam_a = copy.deepcopy(opt.state_dict())
    rng_a = copy.deepcopy(rng.bit_generator.state)
    coverage_a = copy.deepcopy(coverage)
    cpu_rng_a = torch.get_rng_state().clone()
    cuda_rng_a = torch.cuda.get_rng_state(device).clone()

    restored = runner.restore_safe(cpdir, device, identity)
    rm, ro, rr, rc, meta, changed = restored
    assert meta["step"] == 0 and not changed
    restored_names = {id(p): n for n, p in rm.named_parameters()}
    restored_layout = [(g["block"], g["lr"], [restored_names[id(p)] for p in g["params"]])
                       for g in ro.param_groups]
    assert restored_layout == groups
    assert len({id(p) for g in ro.param_groups for p in g["params"]}) == len(expected_trainable)
    trace_b = runner.train_steps(rm, ro, store, events, rr, rc, schedule[:1], end_step=1)
    for key, value in state_a.items():
        torch.testing.assert_close(rm.state_dict()[key], value, atol=2e-9, rtol=2e-9)
    for group_a, group_b in zip(adam_a["param_groups"], ro.state_dict()["param_groups"]):
        assert {k: v for k, v in group_a.items() if k != "params"} == {k: v for k, v in group_b.items() if k != "params"}
        assert group_a["params"] == group_b["params"]
    for old_state, new_state in zip(adam_a["state"].values(), ro.state_dict()["state"].values()):
        for key, val in old_state.items():
            other = new_state[key]
            if torch.is_tensor(val):
                if key == "step": assert torch.equal(val, other)
                else: torch.testing.assert_close(val.to(other.device), other, atol=2e-9, rtol=2e-9)
    assert rr.bit_generator.state == rng_a and rc == coverage_a
    assert torch.equal(torch.get_rng_state(), cpu_rng_a)
    assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng_a)
    assert trace_a[0]["loss"] == trace_b[0]["loss"] or math.isclose(trace_a[0]["loss"], trace_b[0]["loss"], rel_tol=2e-9, abs_tol=2e-9)

    altered = copy.deepcopy(identity); altered["source"] = "changed"
    _raises(lambda: runner.restore_safe(cpdir, device, altered), "changed source")
    altered_code = copy.deepcopy(identity); altered_code["code"]["synthetic"] = "changed"
    _raises(lambda: runner.restore_safe(cpdir, device, altered_code), "changed code")
    stale_rng = copy.deepcopy(schedule[0]); stale_rng["numpy_rng_after"] = {"stale": True}
    _raises(lambda: runner.train_steps(rm, ro, store, events,
        np.random.default_rng(83102), copy.deepcopy(initial_coverage), [stale_rng], end_step=1,
        update=lambda *args: (_ for _ in ()).throw(AssertionError("stale RNG reached update"))),
        "expired schedule RNG")
    failure_saves = []
    _raises(lambda: runner.train_steps(rm, ro, store, events,
        np.random.default_rng(83102), copy.deepcopy(initial_coverage), schedule[:1], end_step=1,
        save=lambda step, history: failure_saves.append(step),
        update=lambda *args: (_ for _ in ()).throw(FloatingPointError("synthetic update failure"))),
        "failed update")
    assert failure_saves == [0], "failed update published an unsafe continuation checkpoint"
    damage = output / "damaged"
    shutil.copytree(cpdir, damage)
    metadata_path = damage / "latest.json"
    pointer = json.loads(metadata_path.read_text(encoding="utf-8"))
    training_path = damage / pointer["directory"] / "training.pt"
    with training_path.open("ab") as stream: stream.write(b"corruption")
    _raises(lambda: runner.restore_safe(damage, device, identity), "corrupt training state")
    damaged_model = output / "damaged-model"
    shutil.copytree(cpdir, damaged_model)
    pointer = json.loads((damaged_model / "latest.json").read_text(encoding="utf-8"))
    model_path = damaged_model / pointer["directory"] / "model.json"
    with model_path.open("ab") as stream: stream.write(b"corruption")
    _raises(lambda: runner.restore_safe(damaged_model, device, identity), "corrupt model state")
    return {"steps": [1], "checkpoint_step": 0, "adam_groups": len(groups),
            "updates_equal": True, "source_and_corruption_rejected": True}


def _wait_tree_test():
    import ctypes
    import linear_cache_cost_gate_20261019 as cost_runner
    pid_path = Path(tempfile.gettempdir()) / f"fixed-pair-grandchild-{time.time_ns()}.pid"
    code = ("import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c',"
            "'import time; time.sleep(30)'],creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)); "
            "open(sys.argv[1],'w').write(str(p.pid)); time.sleep(30)")
    proc = subprocess.Popen([sys.executable, "-c", code, str(pid_path)],
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        deadline = time.monotonic() + 5
        while not pid_path.exists() and proc.poll() is None and time.monotonic() < deadline:
            time.sleep(.02)
        assert pid_path.exists() and proc.poll() is None
        child_pid = int(pid_path.read_text())
        cleanup = {}
        try: cost_runner.wait_owned(proc, .05, cleanup)
        except subprocess.TimeoutExpired: pass
        else: raise AssertionError("owned timeout did not fire")
        assert proc.poll() is not None and "failure" not in cleanup
        kernel = ctypes.windll.kernel32
        kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel.WaitForSingleObject.restype = ctypes.c_uint32
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x00100000, False, child_pid)
        if handle:
            try: assert kernel.WaitForSingleObject(handle, 5000) == 0
            finally: kernel.CloseHandle(handle)
        else: assert kernel.GetLastError() == 87
        return {"owned_tree_cleaned": True}
    finally:
        if pid_path.exists(): pid_path.unlink()


def _native_receipt_test(output):
    import ctypes
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "recovery_complete.json"
    marker.write_text("{}\n", encoding="utf-8")
    normal_worker = output / "normal-worker.json"
    normal_code = "import json,os,sys; open(sys.argv[1],'w').write(json.dumps({'pid':os.getpid()}))"
    ok = runner.native_job([sys.executable, "-c", normal_code, str(normal_worker)],
                           output / "normal", 10., marker, normal_worker)
    assert ok["exit_code"] == 0 and not ok["timed_out"]
    assert ok["status"] == "COMPLETED"
    assert ok["native_receipt_complete"] and ok["includes_child_cost_once"]
    assert ok["seconds"] < 10 and ok["launcher_pid"] and ok["actual_worker_pid"]
    assert ok["actual_worker_pid"] == json.loads(normal_worker.read_text(encoding="utf-8"))["pid"]
    assert (output / "normal" / "native.json").is_file()

    nonzero_worker = output / "nonzero-worker.json"
    nonzero_code = "import json,os,sys; open(sys.argv[1],'w').write(json.dumps({'pid':os.getpid()})); raise SystemExit(7)"
    nonzero = runner.native_job([sys.executable, "-c", nonzero_code, str(nonzero_worker)],
                                output / "nonzero", 10., worker_json=nonzero_worker)
    assert nonzero["exit_code"] == 7 and not nonzero["timed_out"]
    assert nonzero["status"] == "FAILED"
    assert nonzero["native_receipt_complete"] and nonzero["seconds"] < 10
    assert nonzero["includes_child_cost_once"] and nonzero["actual_worker_pid"]
    assert nonzero["actual_worker_pid"] == json.loads(nonzero_worker.read_text(encoding="utf-8"))["pid"]

    pid_path = output / "sleep-grandchild.pid"
    timeout_worker = output / "timeout-worker.json"
    child_code = ("import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c',"
                  "'import time; time.sleep(30)'],creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)); "
                  "open(sys.argv[1],'w').write(str(p.pid)); "
                  "open(sys.argv[2],'w').write('{\"pid\":'+str(__import__('os').getpid())+'}'); time.sleep(30)")
    timeout = runner.native_job([sys.executable, "-c", child_code, str(pid_path), str(timeout_worker)],
                                output / "timeout", 10., worker_json=timeout_worker)
    assert timeout["status"] == "TIMED_OUT"
    assert timeout["timed_out"] is True and timeout["native_receipt_complete"]
    assert timeout["includes_child_cost_once"] and timeout["seconds"] >= 3.5
    assert timeout["launcher_pid"] and timeout["actual_worker_pid"]
    assert timeout["actual_worker_pid"] == json.loads(timeout_worker.read_text(encoding="utf-8"))["pid"]
    assert "failure" not in timeout["cleanup"]
    assert pid_path.is_file()
    child_pid = int(pid_path.read_text(encoding="utf-8"))
    kernel = ctypes.windll.kernel32
    kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel.WaitForSingleObject.restype = ctypes.c_uint32
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x00100000, False, child_pid)
    if handle:
        try: assert kernel.WaitForSingleObject(handle, 5000) == 0
        finally: kernel.CloseHandle(handle)
    else: assert kernel.GetLastError() == 87
    return {"normal_seconds": ok["seconds"], "nonzero_exit": nonzero["exit_code"],
            "timeout_seconds": timeout["seconds"], "timeout_owned_tree_cleaned": True}


def run_qa(output):
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    import linear_cache_cost_gate_20261019 as cost
    cost.load_runtime()
    result = {"status": "RUNNING", "scope": "synthetic new-loop/checkpoint/metric QA",
              "real_train_read": False, "val_read": False, "test_read": False,
              "S_O_source_read": False}
    try:
        _identity_budget_tests(output)
        _risk_and_unit_tests()
        _metadata_tests()
        result["sampler_checkpoint"] = _sampler_checkpoint_test(output / "synthetic")
        result["owned_tree"] = _wait_tree_test()
        result["native_receipts"] = _native_receipt_test(output / "native")
        result["dependency_hashes"] = runner.dependency_hashes()
        assert "tests/test_fixed_edge_pair_20261019.py" in result["dependency_hashes"]
        result.update(status="PASS", code_sha256=runner.dependency_hashes())
    except BaseException as exc:
        result.update(status="FAIL", error=repr(exc), traceback=traceback.format_exc())
        (output / "qa.json").write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        raise
    (output / "qa.json").write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return result
