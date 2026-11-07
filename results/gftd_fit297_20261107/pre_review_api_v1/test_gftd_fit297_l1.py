"""Algebraic scoring checks only; no physical simulations or event files."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import gftd_fit297_l1_20261107 as l1


def event(n=1511):
    return dict(id='mock',group='fixture',label='stable',x=np.zeros((n,30),dtype=np.float64),
        classification_meta=dict(equilibrium_delta=np.zeros(10),inertia=np.ones(10),omega_s=2*np.pi*60,event_time_s=None))


def test_point_error_and_pooled_physical_components():
    truth=np.zeros((3,30));pred=truth.copy();pred[:,10:20]=2.;pred[:,20:30]=3.
    row=l1.error_metrics(pred,truth,np.ones(10),np.full(10,2.))
    assert row['mean_point_error']==pytest.approx((2/l1.S_OMEGA)**2+2.25)
    assert row['common_omega_rmse']==pytest.approx(2.)
    assert row['normalized_p_rmse']==pytest.approx(1.5)
    assert row['coi_normalized_rmse']==pytest.approx(0.)
    # Event risk uses equal event weights, independently of duration.
    assert l1.risk([1.,3.])==pytest.approx(.8*2+.2*3)


def test_exact_alignment_same_point_and_actual_short_interval():
    e=event(31)
    fine=np.zeros((3001,50));coarse=np.zeros((1501,50))
    # pred[1] is paired to x[11], never the initialization x[10].
    e['x'][11:31,20:30]=4.
    fine[2:42:2,20:30]=4.;coarse[1:21,20:30]=4.
    row=l1.score_event(e,fine,coarse,delta_star=np.zeros(10),mass=np.ones(10),p_scale=np.ones(10))
    assert row['actual_future_points']==20
    assert row['scopes']['actual15']['fine']['mean_point_error']==0.
    assert row['scopes']['actual15']['fine']['n']==20
    assert row['scopes']['1']['available'] is False
    assert row['scopes']['actual15']['numerically_admissible'] is True


def test_fine_internal_latent_nan_fails_numeric_gate_even_if_decimation_finite():
    e=event();fine=np.zeros((3001,50));coarse=np.zeros((1501,50))
    fine[1,49]=np.nan
    row=l1.score_event(e,fine,coarse,delta_star=np.zeros(10),mass=np.ones(10),p_scale=np.ones(10))
    assert not row['scopes']['1']['numerically_admissible']
    assert not row['scopes']['8']['finite']
    assert row['classification_8s']['classification']['category']=='numerical_unresolved'


def test_short_path_rejected_and_common_angle_gauge_allowed():
    e=event();fine=np.zeros((3001,50));coarse=np.zeros((1501,50))
    with pytest.raises(ValueError,match='shape mismatch'):
        l1.score_event(e,fine[:-1],coarse,delta_star=np.zeros(10),mass=np.ones(10),p_scale=np.ones(10))
    e['classification_meta']['equilibrium_delta']=np.ones(10)*.2
    row=l1.score_event(e,fine,coarse,delta_star=np.zeros(10),mass=np.ones(10),p_scale=np.ones(10))
    assert row['scopes']['8']['fine']['mean_point_error']==0
    e['classification_meta']['equilibrium_delta'][0]+=.01
    with pytest.raises(ValueError,match='gauge'):
        l1.score_event(e,fine,coarse,delta_star=np.zeros(10),mass=np.ones(10),p_scale=np.ones(10))


def test_missing_failure_never_changes_original_denominators():
    e=event();records=[l1.failure_event(e,'initialization',ValueError('mock failure'))]
    summary=l1.summarize(records)
    assert summary['denominators']==dict(all=297,stable=267,unstable=30,initializations=594)
    assert summary['scopes']['stable_5']['official_fine'] is None
    assert summary['scopes']['stable_5']['unresolved_ids']==['mock']
    assert summary['classifications']['8s']['failure_ids']==['mock']


def test_risk_perturbation_and_paired_identity_gate():
    e=event();fine=np.zeros((3001,50));coarse=np.zeros((1501,50))
    row=l1.score_event(e,fine,coarse,delta_star=np.zeros(10),mass=np.ones(10),p_scale=np.ones(10))
    result=l1.paired_summary([row],[row])
    assert result['differences']['stable_5'] is None
    with pytest.raises(ValueError,match='identities'):
        l1.paired_summary([row],[dict(row,event_id='other')])
    with pytest.raises(ValueError,match='duplicate'):
        l1.summarize([row,row])


def test_unresolved_and_latency_are_separate_from_resolved_misses_and_timing():
    e=event();e['label']='unstable'
    raw=dict(category='latency_uncovered',predicted_los=False,window_settled=False,trigger_index=None)
    latency=l1._classification({'fine_classification':raw},e,False,800)
    assert latency['classification']['category']=='latency_uncovered'
    unresolved=l1._classification({'fine_classification':dict(raw,category='window_no_los_unsettled')},e,False,800)
    resolved=l1._classification({'fine_classification':dict(raw,category='window_no_los_unsettled')},e,True,800)
    records=[dict(event_id=name,label='unstable',scopes={},classification_8s=c) for name,c in
             (('latency',latency),('unresolved',unresolved),('resolved',resolved))]
    counts=l1.summarize(records)['classifications']['8s']
    assert counts['missed_unstable_ids']==['resolved']
    assert counts['numerical_unresolved_ids']==['unresolved']
    assert counts['uncovered_ids']==['latency']
    assert counts['unsettled_ids']==['resolved']
    # A known true trigger outside this scoring window does not supply timing T.
    e['classification_meta']['event_time_s']=9.
    hit=dict(category='predicted_los',predicted_los=True,window_settled=False,trigger_index=700)
    eight=l1._classification({'fine_classification':hit},e,True,800)
    actual=l1._classification({'fine_classification':hit},e,True,1500)
    assert eight['trigger_time_error_s'] is None
    assert actual['trigger_time_error_s']==pytest.approx(-2.)
