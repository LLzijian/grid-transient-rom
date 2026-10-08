"""Synthetic contracts for Loop0085; no cached or learned model is loaded."""
import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
spec=importlib.util.spec_from_file_location('guard0085',ROOT/'experiments'/'audit_lyapunov_guard_0085.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def test_analytic_guard_alpha_p_residual_and_decay_slack():
    A0=-np.eye(32);delta=2*np.eye(32);out=mod.guard(A0,delta)
    assert out['alpha']==pytest.approx(.245)
    assert out['P_residual']<1e-14 and out['P_eigenvalues'].min()>1e-10
    actual=A0+out['alpha']*delta
    assert mod.local_qa(actual,out['P'])['decay_max_eigenvalue']==pytest.approx(-.51,abs=1e-12)


def test_small_mu_keeps_full_registered_update_and_invalid_p_is_rejected():
    out=mod.guard(-np.eye(32),.4*np.eye(32))
    assert out['mu']==pytest.approx(.4) and out['alpha']==1.
    with pytest.raises(AssertionError): mod.guard(np.eye(32),np.zeros((32,32)))


def test_guard_and_local_qa_reject_nonfinite_and_wrong_shape_inputs():
    with pytest.raises(ValueError): mod.guard(np.eye(31),np.eye(31))
    bad=-np.eye(32);bad[0,0]=np.nan
    with pytest.raises(ValueError): mod.guard(bad,np.eye(32))
    with pytest.raises(ValueError): mod.local_qa(np.eye(32),np.eye(31))


def test_guarded_theta_is_exact_24_amplitude_interpolation():
    theta0=np.linspace(-.2,.2,24);proposal=theta0+np.linspace(.01,.24,24);alpha=.245
    structured=SimpleNamespace(storage_gamma=torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(value,dtype=torch.float64)) for value in theta0[:8]]),port_coefficient=torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(theta0[8:16])),torch.nn.Parameter(torch.tensor(theta0[16:24]))]))
    model=SimpleNamespace(structured=structured);guarded=theta0+alpha*(proposal-theta0)
    mod.set_theta(model,guarded)
    assert np.array_equal(mod._theta(model),guarded)


def test_conditional_residual_is_exact_affine_interpolation_with_preserved_coverage():
    counts=np.full(346,134);counts[:150]+=1;event=np.repeat(np.arange(346),counts);n=len(event)
    labels=np.array(['stable']*311+['unstable']*35);start=np.ones(n,dtype=int);start[np.r_[0,np.cumsum(counts)[:-1]]]=10
    table={'event_index':event,'label':labels[event],'group':np.array([f'g{i%24}' for i in event]),'start':start,
           'residual_baseline_fine':np.zeros((n,31)),'residual_candidate_fine':np.full((n,31),2.),'residual_baseline_coarse':np.zeros((n,31)),'residual_candidate_coarse':np.full((n,31),4.)}
    summary,arrays=mod.conditional(table,.25)
    assert np.allclose(arrays['guard_fine'],.5) and np.allclose(arrays['guard_coarse'],1.)
    assert summary['guard']['overall']['total']==pytest.approx(np.sqrt(31*.25))
    assert arrays['numerical'].shape==(46514,)
