import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from screen_structure_0078 import screens, select_extensions


def test_common5_distinguishes_missing_truth_from_failed_eligible_forecast():
    from screen_structure_0078 import _common5
    complete = {'horizon':'5', 'n':500, 'sse':125., 'complete':True}
    short = {'horizon':'5', 'n':0, 'sse':None, 'complete':False}
    failed = {'horizon':'5', 'n':500, 'sse':None, 'complete':True}
    assert _common5({'common_frequency':[complete,short]}) == .5
    assert _common5({'common_frequency':[complete,short,failed]}) is None


def summary(v):
    return {h: {"mean": v, "pooled": v, "cvar10": v} for h in ("1.0", "2.0", "5.0")}


def evidence(v=1.0):
    return {"summary": summary(v), "unstable_available_prefix": {"pooled": v, "cvar10": v}, "common_frequency": [{"horizon": "5.0", "sse": v * 10, "n": 10}], "l1_numerically_gated_counts": {"detected_unstable": 10, "false_los": 0, "false_window_settled": 2}, "model_numerical_gate_pass": True}


def test_screen_passes_exact_boundary():
    out = screens(evidence(.9), evidence(), evidence())
    assert out["screen_pass"]


def test_nan_is_not_omitted_and_fails():
    bad = evidence(); bad["summary"]["5.0"]["pooled"] = float("nan")
    out = screens(bad, evidence(), evidence())
    assert not out["gates"]["S_5.0_pooled"]
    assert not out["screen_pass"]


def test_extension_requires_both_fixed_improvements_and_sorts():
    arms = {"a": {0: {"risk": 1}, 800: {"risk": .9}, 1200: {"risk": .8}}, "b": {0: {"risk": 1}, 800: {"risk": .9}, 1200: {"risk": .7}}, "bad": {0: {"risk": 1}, 800: {"risk": .9}, 1200: {"risk": .95}}}
    assert select_extensions(arms) == ["b", "a"]


def test_missing_extension_or_nonfinite_fails():
    assert select_extensions({"a": {0: {"risk": 1}, 800: {"risk": .9}, 1200: {"risk": math.nan}}}) == []


def test_actual_numeric_horizon_keys_and_zero_reference_are_safe():
    candidate = evidence(.0); baseline = evidence(0.0)
    candidate["summary"] = {k: candidate["summary"][f"{float(k):.1f}"] for k in ("1", "2", "5")}
    baseline["summary"] = {k: baseline["summary"][f"{float(k):.1f}"] for k in ("1", "2", "5")}
    candidate["common_frequency"][0]["horizon"] = "5"
    baseline["common_frequency"][0]["horizon"] = "5"
    out = screens(candidate, baseline, baseline)
    assert out["gates"]["S_5.0_pooled"]


def test_null_l1_is_a_failed_gate_not_an_exception():
    bad = evidence(); bad["l1_numerically_gated_counts"] = None
    out = screens(bad, evidence(), evidence())
    assert not out["gates"]["L1_detect"]
