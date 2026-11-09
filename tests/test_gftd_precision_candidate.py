from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import multiprocessing
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiments.gftd_initialization_precision_20261109 import candidate
from kiki.models import GFTDKuramotoModel, gftd_fit297


def old_model():
    model = GFTDKuramotoModel(fit_mode="static")
    model.initialization_mode = "causal_two_stage"
    request = dict(format=gftd_fit297.FIT_FORMAT, protocol_sha256=gftd_fit297.PROTOCOL_SHA256,
                   adaptation_indices=[0], input_scale_delta=[1.] * 10, input_scale_omega=[1.] * 10,
                   rstar=[.25] * 10, k_ref=1., source_sha256={"artificial_fixture": "a" * 64})
    model.inference_config = gftd_fit297.make_config(model.functional, "static", request, 1, "fixture")
    return model


def fixture_worker(root, refs):
    candidate.diagnostic.ROOT = Path(root)
    candidate.setup_worker(refs)


def test_spawned_worker_runs_complete_initializer_once_and_reuses_pilot_bytes(tmp_path, monkeypatch):
    diagnostic = candidate.diagnostic
    monkeypatch.setattr(diagnostic, "ROOT", tmp_path)
    for key in candidate.THREADS:
        monkeypatch.setenv(key, "1")
    models = candidate.checkpoint("pilot", {"static": old_model()}, None)
    packet = dict(theta_history=np.zeros((11, 10)), omega_history=np.zeros((11, 10)),
                  history_times=np.linspace(-.1, 0., 11), p0=np.zeros(10))
    row = dict(mode="static", index=0, event_id="artificial", label="stable", status="blocked_by_stage1", old_gates=None)
    job = ("pilot", "fixture", "initialize", row, packet, 0.)
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                             initializer=fixture_worker, initargs=(tmp_path, models)) as pool:
        name, ref, result = next(candidate.execute(pool, iter([job]), 1))
        with pytest.raises(FileExistsError):
            pool.submit(candidate.worker, *job).result()
    assert result["passed"] and result["final_gates"]["passed"] and result["p0_exact"]
    counts = result["counts"]
    assert counts["actual_residual_calls"] > 0
    assert counts["actual_rhs_calls"] == 12 * counts["fixed_dp8_steps_completed"]
    details = json.loads(gzip.decompress(diagnostic.read_verified(
        result["diagnostics"]["path"], result["diagnostics"]["sha256"])))
    assert details["stage1"]["initial"] == [0.] * 10
    assert details["stage2"] is not None
    pilot = dict(action_id="fixture", models=models, tasks={"initialize": {name: ref}})
    refs, restored = candidate.reuse_pilot(pilot, {name: row}, {"artificial": packet})
    assert refs == {name: ref} and restored == {name: result}
    assert candidate.checkpoint("full", {"static": old_model()}, pilot) == models


def test_continuation_uses_saved_fine_prefix_even_if_later_failure(monkeypatch):
    f = old_model().functional
    state = np.zeros(50)
    calls = []

    def fixed(functional, initial, duration, step):
        calls.append((duration, step))
        path = np.zeros((1501, 50))
        last = 2. if step == .005 else 15.
        path[round(last * 100) + 1:] = np.nan
        return candidate.diagnostic.Integration(path, state, None, dict(
            completed=last == 15., step=step, last_accepted_time=last, actual_rhs_calls=12,
            failure=None,
        ))

    def radau(functional, initial, duration, **options):
        assert options["rhs_limit"] == 20000 and options["rtol"] == 1e-11
        return candidate.diagnostic.Integration(np.zeros((101, 50)), state, None,
                                                dict(completed=True, actual_rhs_calls=37))

    monkeypatch.setattr(candidate.fixed_state, "integrate", fixed)
    monkeypatch.setattr(candidate.diagnostic, "integrate", radau)
    result, _ = candidate.continuation(f, state, True)
    assert calls == [(15., .01), (15., .005)]
    assert result["differences"]["2"] is not None and result["differences"]["5"] is None
    assert result["single_radau_reference"]["status"] == "confirmed"
    assert result["counts"]["actual_rhs_calls"] == 61


def test_continuation_leaves_radau_unresolved_when_fine_never_reaches_one_second(monkeypatch):
    state = np.zeros(50)
    path = np.full((1501, 50), np.nan)
    path[:40] = 0.
    monkeypatch.setattr(candidate.fixed_state, "integrate", lambda f, x, duration, step:
                        candidate.diagnostic.Integration(path, state, None, dict(
                            completed=False, step=step, last_accepted_time=.39, actual_rhs_calls=480, failure=None)))
    monkeypatch.setattr(candidate.diagnostic, "integrate", lambda *args, **kw: pytest.fail("no eligible fine prefix"))
    result, _ = candidate.continuation(old_model().functional, state, True)
    assert result["single_radau_reference"]["status"] == "reference_unresolved"
    assert not result["single_radau_reference"]["attempted"]
    assert result["differences"]["1"] is None


def test_pilot_keys_deduplicate_stage1_controls_and_guard_opportunity():
    originals = [dict(mode="static", index=i, role="selected") for i in range(44)]
    controls = [dict(row, role="stage1") for row in originals[:5]]
    assert len(candidate.pilot_keys({"selection": originals + controls})) == 44
    with pytest.raises(ValueError, match="44 unique"):
        candidate.pilot_keys({"selection": originals + controls[:-1] + [dict(mode="modulated", index=0)]})


def test_pilot_gate_keeps_all_five_original_outputs():
    rows = {str(i): dict(mode="static", label="stable") for i in range(6)}
    results = {str(i): dict(old_status="completed", passed=True) for i in range(5)}
    results["5"] = dict(old_status="blocked_by_stage1", passed=True)
    summary = candidate.summarize(rows, results, {}, "pilot")
    assert summary["eligible_for_full"] and summary["added"] == ["5"]
    results["1"]["passed"] = False
    assert not candidate.summarize(rows, results, {}, "pilot")["eligible_for_full"]
