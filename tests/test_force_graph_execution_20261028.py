"""Pure controller checks: all native receipts/jobs here are labeled mocks.

No worker, dataset, torch, optimizer or real native phase is executed.
"""
import importlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import force_graph_execution_20261028 as control

MOCK_COMMIT = "1" * 40


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def mock_phase(base, phase, result):
    # This is temporary unit-test input, never an experimental native receipt.
    write(base / "binding.json", {"files": {"mock-source.py": "mock-sha"}})
    output = base / control.PHASE_OUTPUTS[phase]
    write(output, result)
    inputs = {str(base / "binding.json"): control.sha(base / "binding.json")}
    admission = base / "ownership" / phase / "admission.json"
    write(admission, {"source_commit": MOCK_COMMIT, "phase": phase, "inputs": inputs})
    source = {"mock-source.py": "mock-sha", **inputs, str(admission): control.sha(admission)}
    write(base / "native" / phase / "receipt.json", {
        "MOCK_ONLY": True, "source_commit": MOCK_COMMIT, "phase": phase,
        "source_before": source, "source_after": source})
    write(base / "ownership" / phase / "complete.json", {"status": "COMPLETED", "source_commit": MOCK_COMMIT,
        "phase": phase, "scientific_output_sha256": control.sha(output)})


def mock_clean(value, source_commit, phase):
    assert value["MOCK_ONLY"] is True
    if value["source_commit"] != source_commit or value["phase"] != phase:
        raise ValueError("mock same-source/phase mismatch")
    return value


def journal(base, category, n):
    path = base / "optimizer_calls.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        for _ in range(n):
            for event in ("optimizer_step_attempt", "optimizer_step_completed"):
                stream.write(json.dumps({"MOCK_ONLY": True, "event": event, "category": category}) + "\n")


def mock_calibration(chosen, *, early=False):
    scores = {m: {str(v): {"qualified": chosen[m] == v, "optimizer_attempts": 7 if early and m == "S" else 8,
        "optimizer_completed": 7 if early and m == "S" else 8, "trace": [None] * (7 if early and m == "S" else 8)}
        for v in (1, 3)} for m in control.MODES}
    total = sum(t["optimizer_completed"] for trials in scores.values() for t in trials.values())
    return {"status": "COMPLETED", "actual_optimizer_updates": total, "actual_optimizer_attempts": total,
        "all_trials_processed": True, "multipliers": chosen, "scores": scores}


def test_import_is_data_free_and_commit_format_is_explicit():
    assert "torch" not in sys.modules
    assert control.commit_id(MOCK_COMMIT) == MOCK_COMMIT
    for value in (None, 3, "main", "0" * 39, "Z" * 40):
        with pytest.raises(ValueError):
            control.commit_id(value)
    assert control.PHASES == ("prepare", "qualify", "qa", "calibrate", "study", "summary")
    assert control.TOTAL_LIMIT == 1304
    assert sum(control.NORMAL_CATEGORY_LIMITS.values()) + 108 == 1304


def test_output_is_fixed_and_does_not_accept_another_round(monkeypatch):
    with tempfile.TemporaryDirectory(prefix="force-controller-") as folder:
        root = Path(folder).resolve(); monkeypatch.setattr(control, "ROOT", root)
        expected = root / "private_data/force_graph_20261028/study01"
        assert control.checked_base(expected) == expected
        for path in (root / "study02", expected.parent / "study02", root / "private_data/r_memory_20261025/run01_repair01"):
            with pytest.raises(ValueError):
                control.checked_base(path)


@pytest.mark.parametrize("state", ("RUNNING", "REUSED_PID", "UNKNOWN"))
def test_only_exact_retained_gone_old_lock_is_exempt(monkeypatch, state):
    with tempfile.TemporaryDirectory(prefix="force-controller-") as folder:
        root = Path(folder).resolve(); monkeypatch.setattr(control, "ROOT", root)
        base = root / "private_data/force_graph_20261028/study01"
        old = root / control.OLD / "active.lock"
        lock = {name: {"pid": pid, "creation_filetime_utc": ft}
                for name, (pid, ft) in control.OLD_IDENTITIES.items()}
        write(old, lock); digest = control.sha(old); monkeypatch.setattr(control, "OLD_LOCK_SHA", digest)
        recovery = {"status": "INTERRUPTED_RUNTIME", "lock_sha256": digest, "lock_preserved": True,
            "owners": {name: {"pid_decimal": str(pid), "creation_filetime_utc_decimal": str(ft), "identity_state": "GONE"}
                       for name, (pid, ft) in control.OLD_IDENTITIES.items()}}
        write(root / "results/r_memory_20261025/recovery_20261028.json", recovery)
        before = old.read_bytes()
        proof = control.check_locks(base, state=lambda _: "GONE")[-1]
        assert proof["old_lock_preserved"] and proof["old_JobCPU"] is None
        assert old.read_bytes() == before
        with pytest.raises(ValueError):
            control.check_locks(base, state=lambda _: state)
        # The worker may enter only its exact newly owned Job lock.
        own = {"pid": 123, "creation_filetime_utc": 134000000000000001}
        write(base / "active.lock", {"worker": own})
        with pytest.raises(ValueError):
            control.check_locks(base, state=lambda _: "GONE")
        assert control.check_locks(base, state=lambda _: "GONE", owned_identity=own)
        with pytest.raises(ValueError):
            control.check_locks(base, state=lambda _: "GONE", owned_identity={**own, "pid": 124})
        lock["worker"]["creation_filetime_utc"] = str(lock["worker"]["creation_filetime_utc"])
        write(old, lock)
        with pytest.raises(ValueError):
            control.check_locks(base, state=lambda _: "GONE", owned_identity=own)


@pytest.mark.parametrize("result,accepted", (
    ({"status": "COMPLETED", "training_method": "fast", "all_fast_qualified": True}, True),
    ({"status": "COMPLETED", "training_method": "rk4", "all_fast_qualified": False}, True),
    ({"status": "COMPLETED", "training_method": "fast", "all_fast_qualified": False}, False),
    ({"status": "UNRESOLVED", "training_method": "rk4", "all_fast_qualified": False}, False),
    ({"status": "COMPLETED", "training_method": "split_as_rk4", "all_fast_qualified": True}, False)))
def test_one_qualified_method_or_registered_fallback(monkeypatch, result, accepted):
    monkeypatch.setattr(control, "clean_receipt", mock_clean)
    with tempfile.TemporaryDirectory(prefix="force-controller-") as folder:
        base = Path(folder)
        mock_phase(base, "qualify", result)
        if accepted:
            control._completed_phase(base, MOCK_COMMIT, "qualify")
        else:
            with pytest.raises(ValueError):
                control._completed_phase(base, MOCK_COMMIT, "qualify")


def test_native_source_snapshot_must_match_binding(monkeypatch):
    monkeypatch.setattr(control, "clean_receipt", mock_clean)
    with tempfile.TemporaryDirectory(prefix="force-controller-") as folder:
        base = Path(folder)
        mock_phase(base, "prepare", {"val_read": False, "test_read": False})
        control._completed_phase(base, MOCK_COMMIT, "prepare")
        write(base / "binding.json", {"files": {"mock-source.py": "changed"}})
        with pytest.raises(ValueError):
            control._completed_phase(base, MOCK_COMMIT, "prepare")


def test_actual_qa_and_calibration_journals_gate_but_partial_isolation_continues(monkeypatch):
    monkeypatch.setattr(control, "clean_receipt", mock_clean)
    with tempfile.TemporaryDirectory(prefix="force-controller-") as folder:
        base = Path(folder)
        mock_phase(base, "prepare", {"val_read": False, "test_read": False})
        mock_phase(base, "qualify", {"status": "COMPLETED", "training_method": "rk4", "all_fast_qualified": False})
        mock_phase(base, "qa", {"status": "PASS", "actual_optimizer_updates": 108})
        mock_phase(base, "calibrate", mock_calibration({"L": 1, "S": None, "M": 3}, early=True))
        with pytest.raises(ValueError):
            control.phase_inputs(base, MOCK_COMMIT, "study")
        journal(base, "qa", 108); journal(base, "calibration", 46)
        inputs = control.phase_inputs(base, MOCK_COMMIT, "study")
        assert str(base / "calibration/summary.json") in inputs
        # Even all-model isolation is reported rather than replaced by an extra
        # learning rate/source. The study owns the honest no-eligible-model result.
        mock_phase(base, "calibrate", mock_calibration(dict.fromkeys(control.MODES), early=True))
        assert control.phase_inputs(base, MOCK_COMMIT, "study")
        with (base / "optimizer_calls.jsonl").open("a") as stream:
            stream.write(json.dumps({"event": "optimizer_step_attempt", "category": "calibration", "run_id": "MOCK_failed", "step": 8}) + "\n")
        with pytest.raises(ValueError, match="reconciliation"):
            control.phase_inputs(base, MOCK_COMMIT, "study")
        assert control.call_counts(base)["pending_attempts"] == [{"category": "calibration", "run_id": "MOCK_failed", "step": 8}]
        write(base / "qa/summary.json", {"status": "UNRESOLVED", "actual_optimizer_updates": 108})
        with pytest.raises(ValueError):
            control.phase_inputs(base, MOCK_COMMIT, "study")


def test_journal_bounds_and_interrupted_attempts_are_not_relabelled():
    with tempfile.TemporaryDirectory(prefix="force-controller-") as folder:
        base = Path(folder)
        journal(base, "qa", 108); journal(base, "calibration", 48)
        journal(base, "formal", 960); journal(base, "replay", 80)
        result = control.call_counts(base)
        assert result["attempted_total"] == result["completed_total"] == 1196
        assert result["reserved_repair_calls_used"] == 0
        journal(base, "qa", 1)
        with pytest.raises(ValueError):
            control.call_counts(base)


def test_isolated_failed_run_does_not_block_an_independent_run_counter():
    with tempfile.TemporaryDirectory(prefix="force-controller-") as folder:
        base = Path(folder)
        rows = [
            {"event": "optimizer_step_attempt", "category": "formal", "run_id": "MOCK_failed_L", "step": 17},
            {"event": "optimizer_step_attempt", "category": "formal", "run_id": "MOCK_independent_M", "step": 1},
            {"event": "optimizer_step_completed", "category": "formal", "run_id": "MOCK_independent_M", "step": 1}]
        (base / "optimizer_calls.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        result = control.call_counts(base)
        assert result["attempted_total"] == 2 and result["completed_total"] == 1
        assert result["pending_attempts"] == [{"category": "formal", "run_id": "MOCK_failed_L", "step": 17}]
    with tempfile.TemporaryDirectory(prefix="force-controller-") as folder:
        base = Path(folder)
        write(base / "optimizer_calls.jsonl", {})
        (base / "optimizer_calls.jsonl").write_text(json.dumps({"category": "formal", "event": "optimizer_step_attempt"}) + "\n")
        counts = control.call_counts(base)
        assert counts["attempted_total"] == 1 and counts["completed_total"] == 0
        assert counts["pending_attempt_category"] == "formal"
        (base / "optimizer_calls.jsonl").write_text(json.dumps({"category": "formal", "event": "optimizer_step_completed"}) + "\n")
        with pytest.raises(ValueError):
            control.call_counts(base)


@pytest.mark.parametrize("phase", control.PHASES)
def test_exact_study_api_is_called_without_a_hidden_retry(monkeypatch, phase):
    calls = []
    module = SimpleNamespace(**{name: lambda *args, _phase=name, **kwargs: calls.append((_phase, args, kwargs))
                                for name in control.PHASES})
    monkeypatch.setattr(control.importlib, "import_module", lambda name: module)
    control._study_call(Path("MOCK_BASE"), phase)
    assert calls == [(phase, (Path("MOCK_BASE"),), {} if phase in ("prepare", "summary") else {"device": "cuda"})]


def test_pipeline_is_sequential_and_does_not_rerun_or_resume(monkeypatch):
    with tempfile.TemporaryDirectory(prefix="force-controller-") as folder:
        base = Path(folder)
        monkeypatch.setattr(control, "checked_base", lambda value: base)
        actions = []
        monkeypatch.setattr(control, "freeze", lambda commit, target: actions.append("freeze"))
        def stop_at_qa(target, commit, phase):
            actions.append(phase)
            if phase == "qa":
                raise ValueError("MOCK QA did not pass")
        monkeypatch.setattr(control, "launch", stop_at_qa)
        with pytest.raises(ValueError):
            control.run(base, MOCK_COMMIT)
        assert actions == ["freeze", "prepare", "qualify", "qa"]
        assert not (base / "complete.json").exists()
