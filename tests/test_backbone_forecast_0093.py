import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import backbone_forecast_0093 as m


def cache(n=346, steps=5):
    lengths=np.full(n, steps, dtype=int); labels=np.array(["stable"]*311+["unstable"]*35)
    x=np.zeros((n,steps+1,30)); f=np.zeros((n,2*steps+1,33)); c=np.zeros((n,steps+1,33))
    return {"event_ids":np.arange(n),"labels":labels,"lengths":lengths,"truth":x,
            "qenc":np.zeros((n,3)),"mass":np.ones(10),"p_scale":np.ones(10),
            "fine_baseline":f,"coarse_baseline":c}

def test_validate_cache_and_risk():
    c=m.validate_cache(cache()); assert c["lengths"].shape==(346,)
    assert m.risk(np.zeros(346))["risk"] == 0

def test_missing_or_nonfinite_genuine_prefix_fails():
    c=cache(); c["fine_baseline"][0,2,0]=np.nan
    with pytest.raises(FloatingPointError): m.validate_cache(c)

def test_padding_nan_is_allowed_and_score_pair_is_complete():
    c=cache(); c["lengths"][0] = 3; c["truth"][0,4:, :] = np.nan
    c["fine_baseline"][0, 7:, :] = np.nan; c["coarse_baseline"][0, 4:, :] = np.nan
    candidate=np.zeros_like(c["fine_baseline"]); candidate[0, 7:, :] = np.nan
    candidate[:, :11, :30] = .1
    out=m.score_pair(candidate, c["fine_baseline"], c)
    assert len(out["per_event"]) == 1038
    assert out["unstable_5s"]["candidate"]["mean"] > 0
    assert out["unstable_5s"]["candidate"]["worst4"] == pytest.approx(out["unstable_5s"]["candidate"]["mean"])
    assert out["average_risk"]["candidate"] > 0

def test_nonfinite_genuine_truth_fails():
    c=cache(); c["truth"][0,1,0]=np.nan
    with pytest.raises(FloatingPointError): m.validate_cache(c)

def test_point_and_numeric_zero():
    c=cache(); assert np.allclose(m.point_error(np.zeros((2,33)),np.zeros((2,33)),np.ones(10),np.ones(10)),0)
    rows=m.numeric_metrics(c["fine_baseline"],c["coarse_baseline"],c["lengths"],c["mass"],c["p_scale"])
    assert len(rows)==346 and max(v for r in rows for v in r.values())==0

def test_positive_baseline_severe_tail_arithmetic():
    c=cache(); base=c['fine_baseline']; cand=base.copy()
    base[:,:,20:30]=1.
    cand[:,:,20:30]=.5
    cand[-4:,:,20:30]=2.
    out=m.score_pair(cand,base,c)
    assert len(out['per_event'])==1038
    assert out['average_risk']['baseline']==pytest.approx(1.)
    assert out['unstable_5s']['candidate']['mean']==pytest.approx((31*.25+4*4)/35)
    assert out['unstable_5s']['candidate']['worst4']==pytest.approx(4.)
    bad=cache(); bad['qenc']=np.zeros((346,2))
    with pytest.raises(ValueError): m.validate_cache(bad)

def test_checked_rollout_wiring_and_completed_fine_preserved(tmp_path,monkeypatch):
    calls=[]
    def fake(model,states,lengths,dt,**kwargs):
        calls.append((dt,kwargs))
        if dt==.01: raise FloatingPointError('synthetic coarse failure')
        return np.zeros((346,11,33))
    monkeypatch.setattr(m,'masked_rk4',fake)
    counters={'forecast_runs':0}; progress={}
    with pytest.raises(FloatingPointError):
        m.forecast_candidate(object(),cache(),deadline=1e20,output=tmp_path,
                             progress=progress,counters=counters)
    assert counters['forecast_runs']==2
    assert (tmp_path/'completed_fine.npz').exists()
    assert calls[0][1]['partial']==tmp_path/'partial_fine.npz'
    assert calls[1][1]['partial']==tmp_path/'partial_coarse.npz'
    assert calls[0][1]['event_ids'].shape==(346,)
