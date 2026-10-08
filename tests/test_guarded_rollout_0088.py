"""Synthetic resource/provenance contracts for Loop0088; no learned artifact is opened."""
import importlib.util
import sys
from pathlib import Path
import json
import pytest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"experiments"))
spec=importlib.util.spec_from_file_location("rollout0088",ROOT/"experiments"/"train_guarded_rollout_0088.py")
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def test_grant_retains_the_honest_failed_0087_resource_screen():
    assert mod.EXPECTED_ESTIMATE==pytest.approx(2900.4357)
    assert mod.EXPECTED_ESTIMATE>2400 and mod.TRAIN_CAP==3600.


def test_same_input_comparison_allows_only_the_new_entrypoint_hash():
    old={"model":"a","reader":"b","audit0087":"old-source"};current={"model":"a","reader":"b","audit0087":"old-source","audit0088":"new-source"}
    assert mod._same_inputs(old,current)
    assert not mod._same_inputs(old,{**current,"reader":"changed"})


def test_exact_budget_accounting_retains_prior_cost_and_single_new_caps():
    record=mod.grant_record()
    assert record["previous_actual_seconds"]==pytest.approx(35.83270050000283)
    assert record["new_main_cap_seconds"]==pytest.approx(5400.-35.83270050000283)
    assert record["new_training_cap_seconds"]==3600. and record["forecast_cap_seconds"]==1200.


def test_preflight_validator_accepts_full_precision_timing_not_rounded_literals(tmp_path,monkeypatch):
    times=[3.8601029999990715,6.989050300000211,16.6552037000007];estimate=100*sum(times)+150
    hashes={"model":"x","audit0087":"old"};payload={"completed":True,"authorized":False,"traces":[{"horizon":h,"seconds":t} for h,t in zip((1.,2.,5.),times)],"estimate_seconds":estimate,"hashes_before":hashes,"hashes_after":hashes}
    (tmp_path/"evidence.json").write_text(json.dumps(payload));original=mod.sha
    monkeypatch.setattr(mod,"sha",lambda path:mod.PRE_SHA if path.name=="evidence.json" else mod.SOURCE_SHA)
    assert mod.validate_preflight(tmp_path,{**hashes,"audit0088":"new"})==payload
    monkeypatch.setattr(mod,"sha",original)
