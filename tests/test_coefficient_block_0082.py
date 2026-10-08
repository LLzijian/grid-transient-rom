"""Synthetic contracts for Loop0082; no private data or checkpoints are read."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
spec = importlib.util.spec_from_file_location("coeff0082", ROOT / "experiments" / "fit_coefficient_block_0082.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)


def test_weighted_residual_has_registered_31_rows_and_scaling():
    r=np.zeros((2,33)); r[:,10:20]=np.arange(10); r[:,20:30]=2.; r[:,30:]=np.array([1.,2.,3.])
    L=np.repeat(np.eye(10,3)[None],2,axis=0); mass=np.arange(1.,11.); p=np.full(10,2.)
    out=mod.weighted_residual(r,L,mass,p,1.2613572464007323)
    assert out.shape==(2,31)
    expected=np.sqrt(.1/10.)*np.array([1.,2.,3.])/1.2613572464007323
    assert np.allclose(out[:,-10:][:,:3],expected)
    assert np.allclose(out[:,11:21],1./np.sqrt(10.))


def test_conditional_speed_literal_is_distinct_from_inherited_autonomous_scale():
    assert mod.CONDITIONAL_SPEED_SCALE == 1.003367
    assert mod.AUTONOMOUS_METRIC == (.47675314588683515,1.0033667249602924)
    r=np.zeros((1,33)); r[0,10]=1.; r[0,11]=-.5
    mass=np.arange(1.,11.); L=np.zeros((1,10,3)); out=mod.weighted_residual(r,L,mass,np.ones(10),1.2613572464007323)
    raw=r[0,10:20]; coi=raw-(raw*mass).sum()/mass.sum()
    assert np.allclose(out[0,:10],np.sqrt(.5/10.)*coi/1.003367,atol=1e-15,rtol=0)
    assert out[0,10] == pytest.approx((raw*mass).sum()/mass.sum()/1.003367,abs=1e-15)


def test_columns_are_the_same_linear_metric_as_individual_residuals():
    rng=np.random.default_rng(82); columns=rng.normal(size=(3,33,24)); L=rng.normal(size=(3,10,3)); mass=np.linspace(1,2,10); p=np.linspace(.7,1.3,10)
    got=mod.weighted_columns(columns,L,mass,p,1.2613572464007323)
    expected=np.stack([mod.weighted_residual(columns[:,:,j],L,mass,p,1.2613572464007323) for j in range(24)],-1)
    assert np.allclose(got,expected,atol=1e-14,rtol=0)


def test_simpson_rules_integrate_cubic_exactly_on_one_stored_interval():
    for fine in (False,True):
        times,weights,factor=mod.simpson_rule(fine); values=np.array(times)**3
        assert np.dot(weights,values)*factor == pytest.approx(.01**4/4,abs=1e-18)
    segment=torch.tensor([[[0.],[2.]]],dtype=torch.float64)
    duration=torch.tensor([1])
    assert torch.allclose(mod._linear(segment,duration,.0025),torch.tensor([[.5]],dtype=torch.float64),atol=1e-15,rtol=0)


def test_equal_event_weighting_despite_unequal_window_counts():
    event=np.r_[np.zeros(100,dtype=int),np.arange(1,346)]; w=mod.event_weights(event)
    assert w.sum()==pytest.approx(1.)
    assert w[event==0].sum()==pytest.approx(1/346)
    assert w[event==123].sum()==pytest.approx(1/346)


def test_design_divides_integrals_by_duration_seconds(monkeypatch):
    class Model:
        mass=torch.ones(10,dtype=torch.float64); p_scale=torch.ones(10,dtype=torch.float64); force_scale=torch.tensor(1.2613572464007323,dtype=torch.float64)
        def begin_graph_cache(self): pass
        def end_graph_cache(self): pass
    def fake_integrals(model,x,q,duration,step):
        base=np.zeros((len(duration),33)); base[:,20]=.1
        return base,np.zeros((len(duration),33,24))
    monkeypatch.setattr(mod,"_integrals",fake_integrals)
    table={"event_index":np.arange(346),"start":np.full(346,10),"end":np.full(346,11),"L":np.repeat(np.eye(10,3)[None],346,axis=0),"x":[np.zeros((2,30)) for _ in range(346)],"q":[np.zeros((2,3)) for _ in range(346)]}
    out=mod._design(Model(),table)
    assert out["c"][0,11] == pytest.approx(10/np.sqrt(10),abs=1e-13)


def test_ridge_solution_and_inactive_column_are_finite_and_stationary():
    rng=np.random.default_rng(5); c=rng.normal(size=(4,31)); A=rng.normal(size=(4,31,24)); A[:,:,23]=0.; w=np.array([.1,.2,.3,.4])
    fit=mod.solve_ridge(c,A,w)
    assert not fit['active'][23] and fit['theta_delta'][23]==0
    assert np.linalg.norm(fit['K']@fit['beta']-fit['g']) < 1e-9


def test_registered_component_sums_precede_equal_window_event_aggregation():
    table={"event_index":np.arange(346),"label":["stable"]*311+["unstable"]*35,"group":["g"]*346,"start":np.full(346,10)}
    residual=np.ones((346,31)); summary=mod.conditional_summary(table,residual)
    assert summary["overall"]["total"] == pytest.approx(np.sqrt(31))
    assert summary["overall"]["physical"] == pytest.approx(np.sqrt(21))
    assert summary["overall"]["latent"] == pytest.approx(np.sqrt(10))
    assert len(summary["per_event"]["total"])==346 and len(summary["window_counts"])==346


def test_deadline_guard_fails_before_more_work():
    with pytest.raises(TimeoutError): mod.require_time(-1.,"synthetic")


class Decay(torch.nn.Module):
    def begin_graph_cache(self): pass
    def end_graph_cache(self): pass
    def field(self,z): return -2.*z


def test_checked_rk4_integrates_actual_prediction_against_analytic_decay():
    q0=torch.tensor([[0.,1.,2.]+[.5]*30,[-1.,3.,.5]+[-.2]*30],dtype=torch.float64)
    lengths=np.array([10,3]); out=mod._checked_rollout(Decay(),q0,lengths,.01)
    for row,n in enumerate(lengths):
        truth=q0[row].numpy()*np.exp(-2*np.arange(n+1)[:,None]*.01)
        assert np.max(np.abs(out[row,:n+1]-truth)) < 1e-8
    assert np.isnan(out[1,4:]).all()
