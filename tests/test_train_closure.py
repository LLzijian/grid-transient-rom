"""End-to-end synthetic optimizer/RNG/device continuation, without real data."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src")]
import train_closure as runner
import train_local_memory_0098 as old
from closure_training_runtime import TrainingClosureROM


@pytest.fixture(scope="module", autouse=True)
def match_cli_thread_policy():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_actual_training_resume_matches_uninterrupted(tmp_path, monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    source, events = runner.synthetic_fixture(12)
    source.export(tmp_path / "source.json")
    monkeypatch.setattr(old, "read_train", lambda: (events, []))
    monkeypatch.setattr(runner, "horizon_for_update", lambda step: .03)
    def args(folder, updates, resume=False):
        return argparse.Namespace(device=device, source=tmp_path / "source.json", output=folder,
            updates=updates, max_seconds=120., seed=1001, final=False, checkpoint_every=1,
            resume=resume, allow_device_change=False)
    uninterrupted, resumed = tmp_path / "full", tmp_path / "resumed"
    runner.train(args(uninterrupted, 2))
    runner.train(args(resumed, 1))
    runner.train(args(resumed, 2, True))
    a = TrainingClosureROM.from_export(uninterrupted / "step000002/model.json")
    b = TrainingClosureROM.from_export(resumed / "step000002/model.json")
    for k, v in a.state_dict().items():
        torch.testing.assert_close(v, b.state_dict()[k], rtol=0, atol=1e-13)
    sa = torch.load(uninterrupted / "step000002/training.pt", weights_only=False)
    sb = torch.load(resumed / "step000002/training.pt", weights_only=False)
    assert sa["coverage"] == sb["coverage"] and sa["numpy_rng"] == sb["numpy_rng"]
    assert torch.equal(sa["torch_rng"], sb["torch_rng"])
    if device == "cuda":
        assert torch.equal(sa["cuda_rng"], sb["cuda_rng"])
    for key, va in sa["optimizer"]["state"].items():
        for name, value in va.items():
            torch.testing.assert_close(value.cpu(), sb["optimizer"]["state"][key][name].cpu(), rtol=0, atol=1e-13)
    meta = json.loads((resumed / "step000002/checkpoint.json").read_text())
    assert meta["trace"][0]["sampling"] == json.loads((uninterrupted / "step000002/checkpoint.json").read_text())["trace"][0]["sampling"]
    identity = meta["identity"]
    identity["seed"] = 999
    with pytest.raises(ValueError, match="identity changed"):
        runner.restore_checkpoint(resumed, runner.resolve_device(device), identity)
    assert meta["test_read"] is False and meta["val_read"] is False


def test_checkpoint_tamper_and_changed_device_are_not_silent(tmp_path):
    source, _ = runner.synthetic_fixture(3)
    source.__class__ = TrainingClosureROM
    opt = runner.make_optimizer(source)
    rng = np.random.default_rng(42)
    folder = runner.save_checkpoint(tmp_path, source, opt, rng, {}, 0, {"seed": 42}, [])
    pointer = json.loads((tmp_path / "latest.json").read_text())
    original = (folder / "training.pt").read_bytes()
    (folder / "training.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="file hash changed"):
        runner.restore_checkpoint(tmp_path, torch.device("cpu"), {"seed": 42})
    (folder / "training.pt").write_bytes(original)
    meta = json.loads((folder / "checkpoint.json").read_text())
    meta["environment"]["device"] = "cuda:0"
    runner.write(folder / "checkpoint.json", meta)
    pointer["checkpoint_sha256"] = runner.sha(folder / "checkpoint.json")
    runner.write(tmp_path / "latest.json", pointer)
    with pytest.raises(ValueError, match="environment changed"):
        runner.restore_checkpoint(tmp_path, torch.device("cpu"), {"seed": 42})
    runner.restore_checkpoint(tmp_path, torch.device("cpu"), {"seed": 42}, allow_device_change=True)


def test_real_cpu_checkpoint_migrates_adam_moments_to_cuda(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    source, _ = runner.synthetic_fixture(3)
    source.__class__ = TrainingClosureROM
    opt = runner.make_optimizer(source)
    source.field(torch.ones(2, 40, dtype=torch.float64) * .01).square().mean().backward()
    opt.step()
    runner.save_checkpoint(tmp_path, source, opt, np.random.default_rng(42), {}, 1, {"seed": 42}, [])
    cuda = runner.resolve_device("cuda")
    with pytest.raises(ValueError, match="environment changed"):
        runner.restore_checkpoint(tmp_path, cuda, {"seed": 42})
    model, restored, _, _, _, changed = runner.restore_checkpoint(tmp_path, cuda, {"seed": 42}, True)
    assert changed and model.mass.is_cuda
    for state in restored.state.values():
        assert state["exp_avg"].is_cuda and state["exp_avg_sq"].is_cuda
    with old.graph(model):
        model.field(torch.ones(2, 40, dtype=torch.float64, device=cuda) * .01).square().mean().backward()
    restored.step()


@pytest.mark.parametrize("fault", ["export", "pointer"])
def test_checkpoint_publication_interruptions_do_not_block_resume(tmp_path, monkeypatch, fault):
    model, _ = runner.synthetic_fixture(3)
    model.__class__ = TrainingClosureROM
    opt, rng = runner.make_optimizer(model), np.random.default_rng(42)
    runner.save_checkpoint(tmp_path, model, opt, rng, {}, 0, {"seed": 42}, [])
    old_export, old_write = runner.export_cpu, runner.write
    def fail_export(*args, **kwargs):
        raise KeyboardInterrupt("during export")
    def fail_pointer(path, obj):
        if Path(path).name == "latest.json":
            raise KeyboardInterrupt("before latest pointer")
        old_write(path, obj)
    if fault == "export":
        monkeypatch.setattr(runner, "export_cpu", fail_export)
    else:
        monkeypatch.setattr(runner, "write", fail_pointer)
    with pytest.raises(KeyboardInterrupt):
        runner.save_checkpoint(tmp_path, model, opt, rng, {}, 1, {"seed": 42}, [])
    assert json.loads((tmp_path / "latest.json").read_text())["directory"] == "step000000"
    monkeypatch.setattr(runner, "export_cpu", old_export)
    monkeypatch.setattr(runner, "write", old_write)
    runner.restore_checkpoint(tmp_path, torch.device("cpu"), {"seed": 42})
    runner.save_checkpoint(tmp_path, model, opt, rng, {}, 1, {"seed": 42}, [])
    assert json.loads((tmp_path / "latest.json").read_text())["directory"] == "step000001"
    assert list(tmp_path.glob(".step*.pending-*")) if fault == "export" else list(tmp_path.glob("orphan-*"))


@pytest.mark.parametrize("phase", ["before_update", "updating", "after_publish"])
def test_interruption_recovery_accounts_for_unsaved_updates(tmp_path, monkeypatch, phase):
    source, events = runner.synthetic_fixture(12)
    source.export(tmp_path / "source.json")
    monkeypatch.setattr(old, "read_train", lambda: (events, []))
    monkeypatch.setattr(runner, "horizon_for_update", lambda step: .02)
    def args(folder, resume=False):
        return argparse.Namespace(device="cpu", source=tmp_path / "source.json", output=folder,
            updates=2, max_seconds=120., seed=11, final=False, checkpoint_every=1 if phase == "after_publish" else 128,
            resume=resume, allow_device_change=False)
    full, interrupted = tmp_path / "full", tmp_path / "interrupted"
    runner.train(args(full))
    original_objective, original_step, original_save = runner.objective, torch.optim.Adam.step, runner.save_checkpoint
    count = [0]
    def failed_objective(*a, **kw):
        count[0] += 1
        if count[0] == 2:
            raise KeyboardInterrupt("after sampling, before updating")
        return original_objective(*a, **kw)
    def failed_step(self, *a, **kw):
        result = original_step(self, *a, **kw)
        count[0] += 1
        if count[0] == 2:
            raise KeyboardInterrupt("Adam completed or partially updated; treat as unsafe")
        return result
    def failed_save(*a, **kw):
        result = original_save(*a, **kw)
        if a[5] == 1:
            raise KeyboardInterrupt("published pointer before saved_step assignment")
        return result
    monkeypatch.setattr(runner, "objective", failed_objective if phase == "before_update" else original_objective)
    monkeypatch.setattr(torch.optim.Adam, "step", failed_step if phase == "updating" else original_step)
    monkeypatch.setattr(runner, "save_checkpoint", failed_save if phase == "after_publish" else original_save)
    with pytest.raises(KeyboardInterrupt):
        runner.train(args(interrupted))
    failure = json.loads((interrupted / "failure.json").read_text())
    assert failure["phase"] == ("complete" if phase == "after_publish" else phase)
    assert failure["last_valid_saved_step"] == (0 if phase == "updating" else 1)
    assert failure["rollback_completed_updates"] == (1 if phase == "updating" else 0)
    monkeypatch.setattr(runner, "objective", original_objective)
    monkeypatch.setattr(torch.optim.Adam, "step", original_step)
    monkeypatch.setattr(runner, "save_checkpoint", original_save)
    runner.train(args(interrupted, resume=True))
    a = TrainingClosureROM.from_export(full / "step000002/model.json")
    b = TrainingClosureROM.from_export(interrupted / "step000002/model.json")
    for key, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=1e-13)
