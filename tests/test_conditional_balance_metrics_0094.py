import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import conditional_balance_metrics_0094 as m

def fixture():
    n=346; lengths=np.full(n,5,int); truth=np.zeros((n,501,30)); fine=np.zeros((n,11,33)); coarse=np.zeros((n,6,33))
    labels=np.array(["stable"]*311+["unstable"]*35); ids=np.arange(n); mass=np.ones(10)
    # opposite signed k/d components create nonzero negative cross terms.
    fine[:,1:,13:23]=1.; fine[:,1:,23:33]=-0.25; coarse[:,1:,13:23]=1.; coarse[:,1:,23:33]=-0.25
    return fine,coarse,truth,lengths,mass,ids,labels

def test_shapes_scores_terms_and_severe_rows():
    f,c,t,l,mass,ids,labels=fixture(); out=m.compute_metrics(f,c,t,l,mass,event_ids=ids,labels=labels)
    assert out["scores"].shape==(2,4,3,346); assert out["terms"].shape==(2,3,346,6)
    assert out["numeric_energy"].shape==(4,3,346); assert out["fullprefixrms"].shape==(4,346)
    assert out["summary"]["grids"]["fine"]["full"]["5"]["mean"] > 0
    assert out["summary"]["grids"]["fine"]["full"]["5"]["worst35"] == pytest.approx(out["summary"]["grids"]["fine"]["full"]["5"]["mean"])

def test_quadratic_terms_reconstruct_full_score_and_padding_is_allowed():
    f,c,t,l,mass,ids,labels=fixture(); l[0]=3; f[0,7:]=np.nan; c[0,4:]=np.nan; t[0,4:]=np.nan; out=m.compute_metrics(f,c,t,l,mass,event_ids=ids,labels=labels)
    for gi in range(2):
        for hi in range(3): assert np.allclose(out["terms"][gi,hi,:, :].sum(-1), out["scores"][gi,3,hi], rtol=1e-10, atol=1e-10)

def test_genuine_nonfinite_and_small_denominator_fail_closed():
    f,c,t,l,mass,ids,labels=fixture(); t[0,1,0]=np.nan
    with pytest.raises(FloatingPointError): m.compute_metrics(f,c,t,l,mass,event_ids=ids,labels=labels)
    f,c,t,l,mass,ids,labels=fixture(); out=m.compute_metrics(f*0,c*0,t,l,mass,event_ids=ids,labels=labels)
    assert not out["summary"]["primary_joint_gate"]
    assert out['summary']['grid_classification']=='inconclusive'
    assert out['summary']['ratios']['fine']['average_full_over_base'] is None


def test_common_speed_norm_is_one_not_one_eleventh_and_severe_is_distinct():
    f,c,t,l,mass,ids,labels=fixture()
    f[:]=0; c[:]=0
    amplitude=np.ones(346); amplitude[-35:]=2.; amplitude[-4:]=3.
    f[:,1:,3:13]=amplitude[:,None,None]*m.SOMEGA
    c[:,1:,3:13]=amplitude[:,None,None]*m.SOMEGA
    out=m.compute_metrics(f,c,t,l,mass,event_ids=ids,labels=labels)
    np.testing.assert_allclose(out['scores'][0,0,0],amplitude**2)
    gs=out['summary']['grids']['fine']['base']
    assert gs['severe_mean']==pytest.approx((31*4+4*9)/35)
    assert gs['severe_worst4']==pytest.approx(9.)
    assert gs['5']['mean']==pytest.approx((311+31*4+4*9)/346)
    assert out['summary']['grid_classification']=='rejected'


def test_weighted_relative_mode_and_negative_cross_terms():
    f,c,t,l,mass,ids,labels=fixture(); mass=np.arange(1.,11.)
    vector=np.arange(10.)-4.; common=(vector*mass).sum()/mass.sum()
    expect=.5*np.mean((vector-common)**2)/m.SOMEGA**2+common**2/m.SOMEGA**2
    np.testing.assert_allclose(m._energy(m._weighted(vector[None],mass)),expect)
    out=m.compute_metrics(f,c,t,l,mass,event_ids=ids,labels=labels)
    assert np.all(out['terms'][...,5]<0)
    np.testing.assert_allclose(out['terms'].sum(-1),out['scores'][:,3],atol=1e-14)


def test_tail_weighted_numeric_denominator_not_plain_mean():
    f,c,t,l,mass,ids,labels=fixture(); f[:]=0; c[:]=0
    f[:,1:,3:13]=.001*m.SOMEGA; c[:,1:,3:13]=.001*m.SOMEGA
    c[0,1:,3:13]+=.001*m.SOMEGA
    out=m.compute_metrics(f,c,t,l,mass,event_ids=ids,labels=labels)
    den=out['summary']['denominator']['fine']['average']
    assert den['numeric']==pytest.approx(1e-6*(.8/346+.2/35))
    np.testing.assert_allclose(out['fullprefixrms'][0,0],.001)


def test_invalid_rms_suppresses_all_ratios():
    f,c,t,l,mass,ids,labels=fixture()
    f[:,1:,3:13]=m.SOMEGA; c[:,1:,3:13]=1.01*m.SOMEGA
    out=m.compute_metrics(f,c,t,l,mass,event_ids=ids,labels=labels)
    assert not out['summary']['numeric_gate']
    assert out['summary']['grid_classification']=='inconclusive'
    assert all(v is None for v in out['summary']['ratios']['fine'].values())


def test_grid_disagreement_is_inconclusive_and_initial_sample_excluded():
    f,c,t,l,mass,ids,labels=fixture(); f[:]=0; c[:]=0
    f[:,1:,3:13]=m.SOMEGA; c[:,1:,3:13]=m.SOMEGA
    f[:,1:,13:23]=(np.sqrt(1.25)+1e-5-1)*m.SOMEGA
    c[:,1:,13:23]=(np.sqrt(1.25)-1e-5-1)*m.SOMEGA
    f[:,0,3:]=1e5; c[:,0,3:]=1e5
    out=m.compute_metrics(f,c,t,l,mass,event_ids=ids,labels=labels)
    assert out['summary']['numeric_gate']
    assert out['summary']['grid_support']=={'fine':True,'coarse':False}
    assert out['summary']['grid_classification']=='inconclusive'


def test_missing_identity_is_not_silently_replaced():
    f,c,t,l,mass,ids,labels=fixture()
    with pytest.raises(ValueError):
        m.compute_metrics(f,c,t,l,mass,event_ids=ids,labels=['stable']*346)
