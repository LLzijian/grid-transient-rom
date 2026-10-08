"""Synthetic-only convex and tail contracts for Loop0095."""
import importlib.util,sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('compat0095',ROOT/'experiments'/'audit_backbone_compatibility_0095.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)

def test_relative_and_finite_guards():
    assert m.rel([1.,2.],[1.,2.])==0
    assert m.finite(np.ones(2),3.)
    assert not m.finite(np.array([np.nan]))

def test_tiny_known_convex_clarabel_contract():
    # Synthetic permitted one-solve smoke: the unconstrained minimizer x=2.
    sys.path.insert(0,str((ROOT/'.loop-work/research_deps_0067').resolve()))
    import cvxpy as cp
    x=cp.Variable();problem=cp.Problem(cp.Minimize(cp.square(x-2)),[x>=0])
    problem.solve(solver='CLARABEL',max_iter=200,time_limit=5.,tol_gap_abs=1e-9,tol_gap_rel=1e-9,tol_feas=1e-9,warm_start=False,verbose=False)
    solver=problem._solver_cache['CLARABEL'];info=solver.get_info();solution=solver.get_solution()
    assert problem.status in (cp.OPTIMAL,cp.OPTIMAL_INACCURATE) and x.value==pytest.approx(2.,abs=1e-7)
    assert abs(problem.value-info.cost_primal)<=1e-7 and np.isfinite([solution.obj_val,solution.obj_val_dual,solution.r_prim,solution.r_dual,info.gap_abs,info.gap_rel]).all()

def test_tail_epigraph_has_fixed_all_event_shape():
    sys.path.insert(0,str((ROOT/'.loop-work/research_deps_0067').resolve()))
    import cvxpy as cp
    e=cp.Variable((3,346),nonneg=True);cs,vars=m._tail_constraints(cp,e,np.arange(311,346),{'average_risk':2.,'severe_mean':1.,'severe_worst4':1.})
    assert len(cs)==9 and vars['u'].shape==(3,346) and vars['u_severe'].shape==(35,)
    values=np.tile(np.linspace(0,.5,346),(3,1)); e.value=values
    tau=np.sort(values,axis=1)[:,-35]; vars['tau'].value=tau
    vars['u'].value=np.maximum(values-tau[:,None],0)
    for hi,risk in enumerate(vars['risks']):
        assert risk.value==pytest.approx(.8*values[hi].mean()+.2*np.sort(values[hi])[-35:].mean())
    severe=values[2,311:]; t=np.sort(severe)[-4]
    vars['tau_severe'].value=t; vars['u_severe'].value=np.maximum(severe-t,0)
    assert t+vars['u_severe'].value.sum()/4==pytest.approx(np.sort(severe)[-4:].mean())


def test_postchecks_fullprefix_and_normalized_certificate():
    sys.path.insert(0,str(ROOT/'experiments'))
    G=np.zeros((2,3,346,37,37)); G[...,36,36]=1.
    diff=np.zeros((3,346,37,37)); diff[0,:,36,36]=4e-6;diff[2,:,36,36]=1e-8
    grams={'grams':G,'difference_grams':diff}
    labels=np.array(['stable']*311+['unstable']*35)
    baseline={'grids':{g:{'base':dict(average_risk=1.,severe_mean=1.,severe_worst4=1.)} for g in ('fine','coarse')}}
    # Native relative-gap convention differs; only the registered max(1,|p|,|d|) matters.
    solver=dict(beta=np.zeros(36),v=np.zeros(36),qa_pass=True,status='optimal',direct_objective=.2,
                native=dict(cost_primal=.2,cost_dual=.2-5e-8,gap_abs=5e-8,gap_rel=2.5e-7,res_primal=0.,res_dual=0.))
    out=m._postchecks(grams,solver,labels,baseline,np.ones(26),np.ones(10),np.ones(36))
    assert out['qa_pass'] and out['fullprefix_weighted_rms_max']==pytest.approx(.0001)
    assert out['step_energy'].shape==(3,346)
    solver['native']['res_dual']=1e-3
    assert not m._postchecks(grams,solver,labels,baseline,np.ones(26),np.ones(10),np.ones(36))['qa_pass']
