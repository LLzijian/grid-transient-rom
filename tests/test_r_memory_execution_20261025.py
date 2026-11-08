"""Pure configuration, provenance, ownership, and recovery-admission checks.

No test starts a native Job, reads event data, or executes an optimizer.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments"), str(ROOT / "src")]
from experiments import r_memory_execution_20261025 as execution

COMMIT = "a" * 40


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def identity(pid=10, creation=20):
    return {"pid": pid, "creation_filetime_utc": creation}


def good_receipt(phase="prepare", exception=False):
    return {"source_commit": COMMIT, "phase": phase,
        "status": "FAILED" if exception else "COMPLETED", "cleanup_verified": True,
        "orphans_cleaned": False, "actual_worker_state": "GONE", "job_active_processes": 0,
        "exit_code": int(exception), "worker_exit_code": int(exception), "launcher_exit_code": int(exception),
        "source_before": {"source": "digest"}, "source_after": {"source": "digest"},
        "worker": {**identity(), "logical_executable": str(execution.LOGICAL_PYTHON)}}


def test_registered_configuration_and_typed_identities():
    assert execution.commit_id(COMMIT) == COMMIT
    assert execution.phase_key("fit", "C", 25102, True) == "fit_C_25102_resume1"
    assert execution.phase_key("runtime-normal") == "runtime_normal"
    assert execution.phase_key("preflight") == "preflight"
    for value in ("short", "A" * 40, None):
        with pytest.raises(ValueError):
            execution.commit_id(value)
    for args in (("fit", "D", 25101), ("fit", "A", True), ("qa", "A", 25101)):
        with pytest.raises(ValueError):
            execution.phase_key(*args)
    for value in ({"pid": True, "creation_filetime_utc": 2}, identity(1, False), identity(-1, 2)):
        with pytest.raises(ValueError):
            execution.typed_identity(value)


def legacy_fixture(tmp_path, monkeypatch):
    owners = {key: identity(*values) for key, values in execution.LEGACY_IDENTITIES.items()}
    lock = put(tmp_path / execution.LEGACY_LOCK, owners)
    digest = execution.sha(lock)
    monkeypatch.setattr(execution, "LEGACY_SHA", digest)
    recovery = {"status": "INTERRUPTED_RUNTIME", "lock_sha256": digest, "lock_preserved": True,
        **{key: {**value, "state": "GONE"} for key, value in owners.items()}}
    put(tmp_path / execution.RECOVERY, recovery)
    return lock


def test_exact_legacy_lock_is_retained_and_rechecked(tmp_path, monkeypatch):
    lock = legacy_fixture(tmp_path, monkeypatch)
    original = lock.read_bytes()
    states = []
    admitted = execution.check_locks(root=tmp_path, base=tmp_path / "new", state=lambda item: states.append(item) or "GONE")
    assert len(admitted) == 1 and len(states) == 2
    assert lock.read_bytes() == original
    for state in ("RUNNING", "REUSED_PID", "UNKNOWN"):
        with pytest.raises(ValueError, match="live, unknown, or reused"):
            execution.legacy_lock_allowed(lock, root=tmp_path, state=lambda _: state)
    lock.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="exact retained"):
        execution.legacy_lock_allowed(lock, root=tmp_path, state=lambda _: "GONE")


def test_other_locks_are_never_silently_reconciled(tmp_path):
    unknown = put(tmp_path / "private_data/closure_mechanism_20261019/P1/other/active.lock", {})
    with pytest.raises(ValueError, match="separate root reconciliation"):
        execution.check_locks(root=tmp_path, base=tmp_path / "new", state=lambda _: "GONE")
    assert unknown.exists()


def test_source_closure_reads_only_local_import_sources(tmp_path):
    put(tmp_path / "unrelated.json", {})
    (tmp_path / "experiments").mkdir()
    (tmp_path / "src/pkg").mkdir(parents=True)
    (tmp_path / "experiments/a.py").write_text("import b\nfrom pkg import child\nimport torch\n", encoding="utf-8")
    (tmp_path / "experiments/b.py").write_text("VALUE=1\n", encoding="utf-8")
    (tmp_path / "src/pkg/__init__.py").write_text("from . import other\n", encoding="utf-8")
    (tmp_path / "src/pkg/child.py").write_text("from .other import VALUE\n", encoding="utf-8")
    (tmp_path / "src/pkg/other.py").write_text("VALUE=2\n", encoding="utf-8")
    names = {str(path.relative_to(tmp_path)).replace("\\", "/") for path in execution.source_closure(tmp_path, ("experiments/a.py",))}
    assert names == {"experiments/a.py", "experiments/b.py", "src/pkg/__init__.py", "src/pkg/child.py", "src/pkg/other.py"}
    with pytest.raises(ValueError, match="missing or outside"):
        execution.source_closure(tmp_path, ("missing.py",))


def test_clean_receipts_require_actual_typed_exit_and_same_source():
    assert execution.clean_receipt(good_receipt(), COMMIT, "prepare")["job_active_processes"] == 0
    assert execution.clean_receipt(good_receipt("runtime_exception", True), COMMIT,
        "runtime_exception", allow_exception=True)["status"] == "FAILED"
    for key, value in (("exit_code", False), ("cleanup_verified", False), ("job_active_processes", False),
                       ("orphans_cleaned", True), ("source_after", {"source": "changed"})):
        receipt = good_receipt(); receipt[key] = value
        with pytest.raises(ValueError, match="native phase receipt"):
            execution.clean_receipt(receipt, COMMIT, "prepare")


def test_prepare_requires_two_actual_same_binding_runtime_probes(tmp_path):
    base = tmp_path / "study"
    source = tmp_path / "source.py"; source.write_text("fixture=1", encoding="utf-8")
    binding = put(base / "binding.json", {"files": {str(source): execution.sha(source)}})
    with pytest.raises(FileNotFoundError):
        execution.phase_inputs(base, "prepare", COMMIT)
    expected = {str(source): execution.sha(source), str(binding.resolve()): execution.sha(binding)}
    paths = []
    for name, exception in (("runtime_normal", False), ("runtime_exception", True)):
        receipt = good_receipt(name, exception)
        receipt["source_before"] = dict(expected); receipt["source_after"] = dict(expected)
        paths.append(put(base / "native" / name / "receipt.json", receipt))
    inputs = execution.phase_inputs(base, "prepare", COMMIT)
    assert all(str(path.resolve()) in inputs for path in paths)
    exception = execution.read(paths[1]); exception["exit_code"] = 0
    put(paths[1], exception)
    with pytest.raises(ValueError, match="native phase receipt"):
        execution.phase_inputs(base, "prepare", COMMIT)
    exception["exit_code"] = 1
    exception["source_before"] = {"other_binding": "other"}; exception["source_after"] = {"other_binding": "other"}
    put(paths[1], exception)
    with pytest.raises(ValueError, match="source closure differ"):
        execution.phase_inputs(base, "prepare", COMMIT)


def test_scientific_dispatch_sets_one_thread_before_study_import(tmp_path, monkeypatch):
    observed = []
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda n: observed.append(("threads", n))))
    def import_study(name):
        observed.append(("import", name))
        return SimpleNamespace(prepare=lambda base: observed.append(("prepare", base)) or {"prepared": True})
    monkeypatch.setattr(execution.importlib, "import_module", import_study)
    assert execution._study_call(tmp_path, "prepare", None, None, "cuda", False) == {"prepared": True}
    assert observed == [("threads", 1), ("import", "experiments.r_memory_study_20261025"), ("prepare", tmp_path)]


def scientific_phase_fixture(tmp_path, eligible):
    base = tmp_path / "study"
    binding = put(base / "binding.json", {"files": {}})
    expected = {str(binding.resolve()): execution.sha(binding)}
    for name, exception in (("runtime_normal", False), ("runtime_exception", True)):
        receipt = good_receipt(name, exception)
        receipt["source_before"] = dict(expected); receipt["source_after"] = dict(expected)
        put(base / "native" / name / "receipt.json", receipt)
    for name in ("prepare.json", "schedule.json"):
        put(base / name, {})
    for name in ("fit_events.npz", "no_update_events.npz"):
        (base / name).write_bytes(b"fixture-only-not-an-event-array")
    for name in ("prepare", "preflight", "qa", "calibrate"):
        put(base / "native" / name / "receipt.json", good_receipt(name))
    put(base / "preflight/summary.json", {"status": "PASS" if all(eligible.values()) else "PARTIAL",
        "actual_optimizer_updates": 0, "eligible": eligible})
    put(base / "qa/summary.json", {"status": "PASS", "actual_optimizer_updates": 204})
    put(base / "calibration/selection.json", {"selected": {"A": 1, "B": 3, "C": 3}})
    return base


def test_partial_preflight_preserves_independent_eligible_configuration(tmp_path):
    base = scientific_phase_fixture(tmp_path, {"A": True, "B": False, "C": True})
    assert execution.phase_inputs(base, "qa", COMMIT)
    assert execution.phase_inputs(base, "calibrate", COMMIT)
    assert execution.phase_inputs(base, "fit", COMMIT, "A", 25101)
    with pytest.raises(ValueError, match="isolated by full-objective"):
        execution.phase_inputs(base, "fit", COMMIT, "B", 25101)
    with pytest.raises(ValueError, match="both B/C"):
        execution.phase_inputs(base, "fit", COMMIT, "C", 25101)


def test_paired_bc_requires_joint_calibration_and_all_failed_is_rejected(tmp_path):
    base = scientific_phase_fixture(tmp_path, {"A": True, "B": True, "C": True})
    assert execution.phase_inputs(base, "fit", COMMIT, "B", 25102)
    put(base / "calibration/selection.json", {"selected": {"A": 1, "B": 1, "C": 3}})
    with pytest.raises(ValueError, match="jointly qualified"):
        execution.phase_inputs(base, "fit", COMMIT, "B", 25102)
    put(base / "preflight/summary.json", {"status": "FAILED", "actual_optimizer_updates": 0,
        "eligible": {"A": False, "B": False, "C": False}})
    with pytest.raises(ValueError, match="full-objective preflight required"):
        execution.phase_inputs(base, "qa", COMMIT)


def test_binding_detects_source_changes_without_old_selection(tmp_path, monkeypatch):
    from experiments import uncapped_windows_job_20261021 as job
    source = tmp_path / "source.py"; source.write_text("value=1", encoding="utf-8")
    command = {"executable": "physical", "logical_executable": "logical", "arguments": [],
        "runtime_scheme": "direct_base_venv", "expected_worker_image": "physical"}
    monkeypatch.setattr(execution, "python_command", lambda *args: command)
    binding = {"source_commit": COMMIT, "no_process_quotas": True, "base": str(tmp_path),
        "logical_executable": "logical", "runtime": {"command": {key: value for key, value in command.items() if key != "arguments"}},
        "files": {str(source): execution.sha(source)}}
    assert execution.check_binding(binding, COMMIT, base=tmp_path) == command
    source.write_text("value=2", encoding="utf-8")
    with pytest.raises(ValueError, match="bound source changed"):
        execution.check_binding(binding, COMMIT, base=tmp_path)
    assert job.verify_files


def test_recorded_calls_bound_resume_replay(tmp_path):
    path = tmp_path / "calls.jsonl"
    rows = []
    for _ in range(33):
        rows.extend([{"event": "optimizer_step_attempt"}, {"event": "optimizer_step_completed"}])
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    assert execution.journal_counts(path, 32)["unpersisted_attempts"] == 1
    with pytest.raises(ValueError, match="32-call"):
        execution.journal_counts(path, 0)
    path.write_text('{"event":"optimizer_step_completed"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="ordered"):
        execution.journal_counts(path, 0)


def test_interrupted_lock_preservation_is_exact_and_confined(tmp_path):
    base = tmp_path / "study"; folder = base / "ownership/fit_A_25101_resume1"
    folder.mkdir(parents=True)
    owner = {name: identity(index + 1, index + 100) for index, name in enumerate(("supervisor", "worker", "launcher"))}
    lock = put(base / "active.lock", owner)
    original = lock.read_bytes()
    proof = {"lock_sha256": execution.sha(lock), "owner": owner}
    with pytest.raises(ValueError, match="stay in the new study"):
        execution.preserve_interrupted_lock(base, tmp_path / "outside", proof, state=lambda _: "GONE")
    with pytest.raises(ValueError, match="owner changed"):
        execution.preserve_interrupted_lock(base, folder, proof, state=lambda _: "RUNNING")
    target = execution.preserve_interrupted_lock(base, folder, proof, state=lambda _: "GONE")
    assert target.read_bytes() == original and not lock.exists()


def test_resume_rejects_a_scientific_failure_before_checkpoint_load(tmp_path):
    base = tmp_path / "study"
    put(base / "models/B_25101/failure.json", {"error": "nonfinite"})
    with pytest.raises(ValueError, match="relabeled"):
        execution.resume_proof(base, COMMIT, "B", 25101, {}, state=lambda _: "GONE")


def test_resume_requires_matching_native_owner_and_complete_hash_checked_checkpoint(tmp_path, monkeypatch):
    base = tmp_path / "study"
    key = "fit_B_25101"
    run = base / "models/B_25101"
    folder = run / "checkpoints/step000032"
    model = put(folder / "model.json", {"model": "fixture"})
    training = folder / "training.pt"; training.write_bytes(b"synthetic-checkpoint-fixture")
    payload = {"optimizer": {"state": {1: {"step": 32}}}, "numpy_rng": {}, "numpy_global_rng": [],
        "python_rng": [], "torch_rng": [], "cuda_rng": [], "coverage": {}, "step": 32}
    metadata = {"step": 32, "mode": "B", "identity": {"seed": 25101},
        "model_sha256": execution.sha(model), "training_sha256": execution.sha(training),
        "parameter_layout": [{"block": "fixture"}]}
    checkpoint = put(folder / "checkpoint.json", metadata)
    put(run / "checkpoints/latest.json", {"directory": "step000032", "checkpoint_sha256": execution.sha(checkpoint)})
    journal = run / "optimizer_calls.jsonl"
    rows = [{"event": event} for _ in range(33) for event in ("optimizer_step_attempt", "optimizer_step_completed")]
    journal.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    owner = {name: identity(index + 1, index + 100) for index, name in enumerate(("supervisor", "worker", "launcher"))}
    put(base / "active.lock", owner)
    source = tmp_path / "bound.py"; source.write_text("fixture=1", encoding="utf-8")
    files = {str(source): execution.sha(source)}
    put(base / "native" / key / "started.json", {"supervisor": owner["supervisor"], "source_before": files})
    put(base / "native" / key / "actual_worker.json", {"worker": owner["worker"], "launcher": owner["launcher"]})
    put(base / "ownership" / key / "worker.json", {**owner["worker"], "source_commit": COMMIT, "phase": key})
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=lambda *args, **kwargs: payload))
    proof = execution.resume_proof(base, COMMIT, "B", 25101, files, state=lambda _: "GONE")
    assert proof["step"] == 32 and proof["optimizer_calls"]["unpersisted_attempts"] == 1
    assert proof["job_active_processes"] is None and proof["cleanup_verified"] is None
    with pytest.raises(ValueError, match="live, unknown, or reused"):
        execution.resume_proof(base, COMMIT, "B", 25101, files, state=lambda _: "REUSED_PID")
    training.write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        execution.resume_proof(base, COMMIT, "B", 25101, files, state=lambda _: "GONE")


def test_import_and_configuration_do_not_offer_process_quotas():
    text = Path(execution.__file__).read_text(encoding="utf-8")
    assert "selected_python_command(" not in text and "validate_selection(" not in text
    assert "timeout=" not in text and "max_process" not in text
