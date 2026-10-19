"""One-shot synthetic CUDA QA of the real fixed-edge worker recovery paths.

Authoring must not execute this file. The root owns the sole frozen-source native
process-tree supervised invocation. No scientific data/checkpoint is read.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import types
from contextlib import ExitStack
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "private_data/closure_mechanism_20261019/repair/worker_resume01/legacy/fixed_edge_pair_20261019.py"
LEGACY_COMMIT = "47ba6f341886aeb4cc235d3c25a9b16bc8f4a24b"
EXPECTED_LEGACY = "4412777530c9a40f80641faf431991e2c62f38194c15023c8d6f1417ce4c14f9"
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src"), str(ROOT / "tests")]
SEED, KEY = 6101, "6101_full8"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_legacy(output):
    """Require the exact local historical blob; no fetch or science-file access."""
    if LEGACY.is_file():
        path, raw = LEGACY, LEGACY.read_bytes()
    else:
        proc = subprocess.run(["git", "show", LEGACY_COMMIT + ":experiments/fixed_edge_pair_20261019.py"],
                              cwd=ROOT, capture_output=True, check=False, timeout=15)
        if proc.returncode:
            raise RuntimeError("legacy blob unavailable locally: " + proc.stderr.decode(errors="replace"))
        raw = proc.stdout
        path = Path(output) / "legacy_source/fixed_edge_pair_20261019.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    if hashlib.sha256(raw).hexdigest() != EXPECTED_LEGACY:
        raise AssertionError("legacy source hash changed")
    module = types.ModuleType("fixed_edge_pair_20261019_legacy_qa")
    # Initialize repository constants at the original location, then make the
    # actual frozen blob visible to the real dependency_hashes() implementation.
    module.__file__ = str(ROOT / "experiments/fixed_edge_pair_20261019.py")
    exec(compile(raw, str(path), "exec"), module.__dict__)
    module.__file__ = str(path)
    return module, path


class SyntheticInterruption(RuntimeError):
    """Test-only stop before the next update; no scientific interruption claim."""


def assert_equal(a, b, path="state"):
    """Numerical state tolerance 2e-9; integers and random streams exact."""
    import numpy as np
    import torch
    if torch.is_tensor(a):
        assert torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape, path
        if a.is_floating_point() and not path.endswith("/step"):
            torch.testing.assert_close(a.cpu(), b.cpu(), atol=2e-9, rtol=2e-9, msg=path)
        else:
            assert torch.equal(a.cpu(), b.cpu()), path
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b), path
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            assert_equal(a[key], b[key], path + "/" + str(key))
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            assert_equal(x, y, path + "/" + str(i))
    elif isinstance(a, float):
        import math
        assert math.isfinite(a) and math.isfinite(b), path
        assert a == b if "rng" in path else math.isclose(a, b, abs_tol=2e-9, rel_tol=2e-9), path
    else:
        assert type(a) is type(b) and a == b, path


def checkpoint_hashes(root, step):
    folder = Path(root) / f"step{step:06d}"
    return {name: digest(folder / name) for name in ("model.json", "training.pt", "checkpoint.json")}


class Fixture:
    """Synthetic inputs and bounded execution size; actual save/restore retained."""
    def __init__(self, output, deadline):
        import numpy as np
        import torch
        import linear_cache_cost_gate_20261019 as cache
        from closure_discovery_benchmarks import toy_source
        from symbolic_closure_rom import _BASE_TERMS, term_key
        from train_closure import cohort_identity
        from train_joint_numerical_pair_0100 import next_samples
        cache.load_runtime()
        if not torch.cuda.is_available():
            raise RuntimeError("registered synthetic worker QA requires CUDA")
        self.output, self.deadline = output, deadline
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.source = output / "synthetic_source.json"
        toy_source().export(self.source)
        terms = [copy.deepcopy(_BASE_TERMS[i]) for i in (0, 1, 2, 3, 4, 5, 7, 8)]
        self.terms = {"full8": terms, "delete7": [t for t in terms if term_key(t) != term_key(_BASE_TERMS[2])]}
        data_rng = np.random.default_rng(83101)
        self.events = [{"id": f"worker-qa-{i}", "group": f"g{i}",
                        "label": "unstable" if i == 1 else "stable",
                        "x": data_rng.normal(0, .01, (length, 30)),
                        "torque": data_rng.normal(0, .01, (length, 10))}
                       for i, length in enumerate((26, 29, 33))]
        self.cohort = cohort_identity(self.events)
        self.reference = {"fingerprint_sha256": hashlib.sha256(b"worker-qa-synthetic-system").hexdigest()}
        rng = np.random.default_rng(SEED)
        coverage = {"cover_order": rng.permutation(len(self.events)).tolist(), "cursor": 0}
        rows = []
        for step in range(1, 53):
            samples, sampling = next_samples(self.events, rng, coverage)
            rows.append({"step": step, "horizon_s": 1.0, "sampling": sampling,
                         "numpy_rng_after": copy.deepcopy(rng.bit_generator.state),
                         "long_counts": [min(100, len(e["x"])-11) for e, _ in samples],
                         "local_counts": [min(20, len(e["x"])-i-1) for e, i in samples]})
        self.schedules = {str(SEED): {"rows": rows}}
        self.schedule = output / "synthetic_schedule.json"
        write(self.schedule, self.schedules)
        self.true_update, self.true_fresh, self.true_restore = cache.one_update, cache.fresh_model, cache.restore_cached

    def check_deadline(self):
        if time.perf_counter() >= self.deadline:
            raise TimeoutError("single QA envelope exhausted")

    def source_model(self, device):
        import torch
        from closure_training_runtime import TrainingClosureROM
        torch.manual_seed(4801)
        return TrainingClosureROM.from_export(self.source).to(device)

    def common(self, code):
        return {"synthetic_only": True, "source_sha256": digest(self.source),
                "cohort": {"sha256": self.cohort, "count": len(self.events)}, "stage_code": code,
                "schedule_sha256": digest(self.schedule),
                "global_metadata_sha256": self.reference["fingerprint_sha256"],
                "bound_files": {str(p.resolve()): digest(p) for p in (self.source, self.schedule)}}

    def preflight(self, output, code):
        output.mkdir(parents=True, exist_ok=True)
        common = self.common(code)
        write(output / "stage_identity.json", common)
        write(output / "schedule.json", self.schedules)
        return self.events, .01, self.terms, common, self.schedules, self.reference

    def invoke(self, runner, output, *, endpoint, target=None, stop_after=None, long_case=False):
        """Call real worker(), preserving its complete callback/control flow."""
        import numpy as np
        import torch
        import closure_discovery_data as data
        import closure_mechanism_study as study
        import linear_cache_cost_gate_20261019 as cache
        self.check_deadline()
        if self.deadline-time.perf_counter() <= 9:
            raise TimeoutError("insufficient remaining envelope for worker startup reserve")
        output.mkdir(parents=True, exist_ok=True)
        code = runner.dependency_hashes()
        write(output / "frozen_code.json", code)
        recovery = None
        if target is not None:
            recovery = output / "synthetic_recovery_request.json"
            write(recovery, {"last_completed_update": target, "kind": "SYNTHETIC_QA_ONLY"})
        args = types.SimpleNamespace(output=output, seed=SEED, structure="full8", recovery=recovery,
                                     max_seconds=self.deadline-time.perf_counter())
        checkpoint = output / KEY / "checkpoints"
        attempt = output / KEY / ("attempt_recovery" if recovery else "attempt_initial")
        observation = {"status": "RUNNING", "worker_called": False, "saves": [], "restores": [],
                       "train_calls": [], "updates": [], "marker_during_updates": []}
        live = {}
        real_train, real_save = runner.train_steps, runner.save_safe

        def fresh(*values, **kw):
            model = self.true_fresh(*values, **kw)
            live["model"] = model
            return model

        def restore(*values, **kw):
            result = self.true_restore(*values, **kw)
            live["model"], live["opt"], live["rng"], live["coverage"] = result[:4]
            observation["restores"].append({"step": result[4]["step"], "changed": result[5]})
            return result

        def save(*values, **kw):
            step = values[5]
            before = (checkpoint / "latest.json").read_bytes() if (checkpoint / "latest.json").exists() else None
            try:
                result = real_save(*values, **kw)
            except BaseException as exc:
                observation["saves"].append({"step": step, "published": False, "error": repr(exc)})
                after = (checkpoint / "latest.json").read_bytes() if (checkpoint / "latest.json").exists() else None
                assert before == after
                raise
            pointer = read(checkpoint / "latest.json")
            assert pointer["directory"] == f"step{step:06d}"
            assert pointer["checkpoint_sha256"] == digest(result / "checkpoint.json")
            meta, hashes = read(result / "checkpoint.json"), checkpoint_hashes(checkpoint, step)
            assert meta["step"] == step and meta["model_sha256"] == hashes["model.json"]
            assert meta["training_sha256"] == hashes["training.pt"]
            previous_steps = [r["step"] for r in observation["saves"] if r["published"]]
            if previous_steps:
                assert step > previous_steps[-1]
            observation["saves"].append({"step": step, "published": True, "hashes": hashes})
            return result

        def update(*values):
            self.check_deadline()
            step = live["step"]
            marker = (attempt / "recovery_complete.json").exists()
            if recovery:
                assert marker == (step > target), "marker must follow replay and precede ordinary continuation"
            if stop_after is not None and step > stop_after:
                raise SyntheticInterruption(f"test-only stop before update {step}")
            genuine = not long_case or step in (1, 51, 52)
            result = self.true_update(*values) if genuine else {"loss": 0., "synthetic_no_model_update": True}
            observation["updates"].append({"step": step, "real_synthetic_update": genuine})
            observation["marker_during_updates"].append({"step": step, "exists": marker})
            return result

        def train(*values, **kw):
            # Replay's explicit endpoint is never changed. Only normal worker's
            # default 1024 endpoint is replaced with the bounded fixture endpoint.
            if "end_step" not in kw:
                kw["end_step"] = endpoint
            start = kw.get("start_step", 0)
            observation["train_calls"].append({"start": start, "end": kw["end_step"],
                                               "save_initial": kw.get("save_initial", True)})
            live.update(model=values[0], opt=values[1], rng=values[4], coverage=values[5], step=start)
            actual_update = kw["update"]

            def count_step(*args):
                live["step"] += 1
                return actual_update(*args)

            kw["update"] = count_step
            result = real_train(*values, **kw)
            live["snapshot"] = {"model": copy.deepcopy(values[0].state_dict()),
                                "adam": copy.deepcopy(values[1].state_dict()),
                                "numpy_rng": copy.deepcopy(values[4].bit_generator.state),
                                "coverage": copy.deepcopy(values[5]),
                                "cpu_rng": torch.get_rng_state().clone(),
                                "cuda_rng": torch.cuda.get_rng_state(self.device).clone(),
                                "numpy_global_rng": copy.deepcopy(np.random.get_state())}
            return result

        error = None
        try:
            with ExitStack() as stack:
                for obj, name, value in (
                    (study, "source_model", self.source_model), (study, "SOURCE", self.source),
                    (study, "SOURCE_SHA", digest(self.source)),
                    (data, "_load_reference_fingerprint", lambda: copy.deepcopy(self.reference)),
                    (runner, "preflight", self.preflight), (runner, "train_steps", train),
                    (runner, "save_safe", save), (cache, "fresh_model", fresh),
                    (cache, "restore_cached", restore), (cache, "one_update", update)):
                    stack.enter_context(patch.object(obj, name, value))
                observation["worker_called"] = True
                runner.worker(args)
            observation["status"] = "COMPLETED"
        except BaseException as exc:
            error = exc
            observation.update(status="INTERRUPTED_SYNTHETICALLY" if isinstance(exc, SyntheticInterruption) else "FAILED",
                               error=repr(exc), traceback=traceback.format_exc())
        observation["worker_result"] = read(attempt / "result.json") if (attempt / "result.json").exists() else None
        observation["recovery_marker"] = read(attempt / "recovery_complete.json") if (attempt / "recovery_complete.json").exists() else None
        write(attempt / "qa_observation.json", observation)
        return observation, live, error


def require_success(observation, error):
    if error is not None:
        raise AssertionError("real synthetic worker failed: " + repr(error)) from error
    assert observation["worker_called"] and observation["worker_result"]["status"] == "COMPLETED"


def trace_rows(output, recovery=False):
    path = output / KEY / ("attempt_recovery" if recovery else "attempt_initial") / "trace.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def resume_case(fixture, runner, output, continuous, target, safe_step, endpoint, long_case=False):
    before, _, error = fixture.invoke(runner, output, endpoint=endpoint, stop_after=target, long_case=long_case)
    assert isinstance(error, SyntheticInterruption), before
    assert before["worker_result"]["last_completed_update"] == target
    assert [row["step"] for row in before["saves"] if row["published"]] == ([0, 50] if safe_step == 50 else [0])
    cp = output / KEY / "checkpoints"
    pointer = (cp / "latest.json").read_bytes()
    assert read(cp / "latest.json")["directory"] == f"step{safe_step:06d}"
    zero_hashes, safe_hashes = checkpoint_hashes(cp, 0), checkpoint_hashes(cp, safe_step)
    assert (output / KEY / "attempt_initial/failed_state.pt").is_file()
    assert not (output / KEY / "attempt_initial/recovery_complete.json").exists()
    restored, live, error = fixture.invoke(runner, output, endpoint=endpoint, target=target, long_case=long_case)
    require_success(restored, error)
    assert restored["restores"] == [{"step": safe_step, "changed": False}]
    assert all(row["step"] != 0 for row in restored["saves"])
    assert all(not call["save_initial"] for call in restored["train_calls"])
    marker = restored["recovery_marker"]
    assert marker["status"] == "COMPLETED" and marker["replayed_steps"] == target-safe_step <= 50
    assert 0 <= marker["seconds"] <= 1200
    assert checkpoint_hashes(cp, 0) == zero_hashes and checkpoint_hashes(cp, safe_step) == safe_hashes
    # Endpoints 2 and 52 are not safe boundaries; latest must stay on 0 or 50.
    assert (cp / "latest.json").read_bytes() == pointer
    assert_equal(continuous["snapshot"], live["snapshot"])
    assert_equal(trace_rows(continuous["output"])[safe_step:], trace_rows(output, True), "actual_worker_trace")
    return {"status": "PASS", "safe_step": safe_step, "target": target, "endpoint": endpoint,
            "replayed_steps": marker["replayed_steps"], "initial": before, "recovery": restored,
            "step0_hashes_unchanged": zero_hashes, "model_adam_rng_coverage_trace_equal": True}, live


def rejection_cases(fixture, runner, output, live):
    import numpy as np
    from closure_training_runtime import TensorEventStore
    from train_joint_numerical_pair_0100 import next_samples
    cp = output / KEY / "checkpoints"
    metadata = read(cp / read(cp / "latest.json")["directory"] / "checkpoint.json")
    identity, pointer, results = metadata["identity"], (cp / "latest.json").read_bytes(), {}

    def rejects(label, action):
        try:
            action()
        except (ValueError, FloatingPointError) as exc:
            results[label] = {"rejected": True, "error": repr(exc)}
        else:
            raise AssertionError("expected actual rejection: " + label)

    altered = copy.deepcopy(identity)
    altered["seed"] += 1
    rejects("wrong_identity", lambda: fixture.true_restore(cp, fixture.device, altered))
    for name in ("model.json", "training.pt"):
        damaged = output / ("damaged_" + name.replace(".", "_"))
        shutil.copytree(cp, damaged)
        path = damaged / read(damaged / "latest.json")["directory"] / name
        with path.open("ab") as stream:
            stream.write(b"synthetic corruption")
        rejects("corrupt_" + name, lambda d=damaged: fixture.true_restore(d, fixture.device, identity))
    rejects("unsafe_checkpoint", lambda: runner.save_safe(cp, live["model"], live["opt"], live["rng"],
                                                          live["coverage"], 1, identity, [], safe=False))
    # Failure precisely at a safe boundary must never invoke the save callback.
    # Replay only the genuine sampler here, without performing 49 model updates.
    rng = np.random.default_rng(SEED)
    coverage = {"cover_order": rng.permutation(len(fixture.events)).tolist(), "cursor": 0}
    for _ in range(49):
        next_samples(fixture.events, rng, coverage)
    saves = []

    def fail_update(*args):
        raise FloatingPointError("synthetic unsafe update at safe boundary 50")

    def forbidden_save(step, trace):
        saves.append(step)
        raise AssertionError("unsafe update reached checkpoint publication")

    rejects("unsafe_update50", lambda: runner.train_steps(live["model"], live["opt"],
        TensorEventStore(fixture.events, fixture.device), fixture.events, rng, coverage,
        fixture.schedules[str(SEED)]["rows"], start_step=49, end_step=50,
        save=forbidden_save, update=fail_update, save_initial=False))
    assert saves == []
    assert (cp / "latest.json").read_bytes() == pointer
    return {"status": "PASS", "results": results, "latest_unchanged": True}


def run(output, max_seconds):
    started = time.perf_counter()
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("sole acceptance invocation requires a new empty output directory")
    if not output.is_relative_to(ROOT):
        raise ValueError("QA output must be inside repository for actual dependency identities")
    output.mkdir(parents=True, exist_ok=True)
    record = {"status": "RUNNING", "pid": os.getpid(), "scope": "synthetic CUDA float64 only",
              "real_data": False, "real_checkpoint": False, "val_read": False, "test_read": False}
    qa = {"status": "RUNNING", "cases": {}, "source_hashes": {},
          "scope_limits": ["long-case steps 2..50 use no-model-update stubs; sampling, trace and saves remain real",
                           "real one_update at short steps 1,2 and long steps 1,51,52",
                           "test-only interruption is not scientific interruption evidence",
                           "outer native supervisor owns process-tree cost and hard timeout"]}
    write(output / "worker.json", record)
    write(output / "qa.json", qa)
    try:
        import fixed_edge_pair_20261019 as runner
        fixture = Fixture(output, started + max_seconds)
        old, legacy_path = load_legacy(output)
        qa["source_hashes"] = {"runner": digest(runner.__file__), "qa": digest(__file__),
                               "legacy": digest(legacy_path), "synthetic_source": digest(fixture.source),
                               "synthetic_schedule": digest(fixture.schedule)}
        old_output = output / "legacy"
        initial, _, error = fixture.invoke(old, old_output, endpoint=2, stop_after=0)
        assert isinstance(error, SyntheticInterruption), initial
        cp = old_output / KEY / "checkpoints"
        hashes, pointer = checkpoint_hashes(cp, 0), (cp / "latest.json").read_bytes()
        negative, _, error = fixture.invoke(old, old_output, endpoint=2, target=0)
        assert isinstance(error, ValueError) and str(error) == "cannot overwrite the active immutable checkpoint", negative
        assert negative["restores"] == [{"step": 0, "changed": False}]
        assert negative["saves"][-1]["step"] == 0 and not negative["saves"][-1]["published"]
        assert checkpoint_hashes(cp, 0) == hashes and (cp / "latest.json").read_bytes() == pointer
        qa["cases"]["legacy_negative_control"] = {"status": "PASS", "expected_actual_rejection": repr(error),
                                                   "source_sha256": EXPECTED_LEGACY, "initial": initial,
                                                   "raw_negative": negative, "step0_unchanged": True}
        write(output / "qa.json", qa)
        fresh_output = output / "continuous2"
        fresh, continuous, error = fixture.invoke(runner, fresh_output, endpoint=2)
        require_success(fresh, error)
        assert continuous["snapshot"]["adam"]["state"], "real synthetic updates must populate Adam state"
        assert [r["step"] for r in fresh["saves"] if r["published"]] == [0]
        assert fresh["recovery_marker"] is None
        continuous["output"] = fresh_output
        qa["cases"]["fresh_step0"] = {"status": "PASS", "observation": fresh}
        for target in (0, 1):
            row, live = resume_case(fixture, runner, output / f"resume_target{target}", continuous, target, 0, 2)
            qa["cases"][f"recovery_target{target}"] = row
            write(output / "qa.json", qa)
        qa["cases"]["identity_corruption_unsafe"] = rejection_cases(fixture, runner, output / "resume_target1", live)
        long_output = output / "continuous52"
        fresh50, continuous50, error = fixture.invoke(runner, long_output, endpoint=52, long_case=True)
        require_success(fresh50, error)
        assert [r["step"] for r in fresh50["saves"] if r["published"]] == [0, 50]
        continuous50["output"] = long_output
        row, _ = resume_case(fixture, runner, output / "resume_step50", continuous50, 51, 50, 52, long_case=True)
        row["continuous"] = fresh50
        qa["cases"]["recovery_step50"] = row
        write(output / "qa.json", qa)
        fixture.check_deadline()
        if fixture.deadline-time.perf_counter() < 40:
            raise TimeoutError("insufficient remaining envelope for fixed native child cases")
        from test_fixed_edge_pair_20261019 import _native_receipt_test
        qa["cases"]["native_receipts"] = {"status": "PASS", **_native_receipt_test(output / "native")}
        fixture.check_deadline()
        qa["status"] = "PASS"
        record.update(status="COMPLETED", qa="PASS")
    except BaseException as exc:
        qa.update(status="FAIL", error=repr(exc), traceback=traceback.format_exc())
        record.update(status="FAILED", error=repr(exc))
        raise
    finally:
        qa["elapsed_process_seconds"] = time.perf_counter()-started
        record["elapsed_process_seconds"] = qa["elapsed_process_seconds"]
        write(output / "qa.json", qa)
        write(output / "worker.json", record)
    return qa


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-seconds", required=True, type=float)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 1200:
        parser.error("single registered QA envelope is at most 1200 seconds")
    qa = run(args.output, args.max_seconds)
    print(json.dumps({"status": qa["status"], "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
