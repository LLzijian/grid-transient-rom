"""Synthetic contracts for Loop0081; no private trajectories are read."""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import pytest

torch=pytest.importorskip('torch')
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'experiments'))
spec=importlib.util.spec_from_file_location('conditional0081',ROOT/'experiments'/'audit_conditional_latent_0081.py')
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)


class Drive(torch.nn.Module):
    def __init__(self, physical=False): super().__init__(); self.physical=physical
    def field(self,z):
        out=torch.zeros_like(z)
        out[:,30:]=z[:,:1].expand(-1,3) if self.physical else 2.
        return out


class AffineQ(torch.nn.Module):
    def field(self,z):
        out=torch.zeros_like(z); out[:,30:]=-2.*z[:,30:]+3.
        return out


def test_rk_stages_use_linear_physical_interpolation():
    segment=torch.zeros((1,11,30),dtype=torch.float64); segment[0,1,0]=2.
    q=mod.integrate_q(Drive(physical=True),segment,torch.tensor([1]),torch.zeros((1,3),dtype=torch.float64),.01)
    # Integral of x(t) from 0 to .01 for x=200t is .01, not either endpoint.
    assert torch.allclose(q,torch.full((1,3),.01,dtype=torch.float64),atol=1e-14,rtol=0)


def test_constant_drive_and_short_window_are_exact_and_finite():
    segment=torch.zeros((2,11,30),dtype=torch.float64)
    out=mod.integrate_q(Drive(),segment,torch.tensor([10,3]),torch.zeros((2,3),dtype=torch.float64),.005)
    assert torch.allclose(out[0],torch.full((3,),.2,dtype=torch.float64),atol=1e-14,rtol=0)
    assert torch.allclose(out[1],torch.full((3,),.06,dtype=torch.float64),atol=1e-14,rtol=0)
    q0=torch.tensor([[.3,-.2,.7],[1.1,.4,-.5]],dtype=torch.float64); duration=torch.tensor([10,3])
    constant=mod.integrate_q(Drive(),segment,duration,q0,.005)
    exact=q0+2.*duration.to(torch.float64)[:,None]*.01
    assert torch.allclose(constant,exact,atol=1e-14,rtol=0)
    L=np.repeat(np.eye(10,3)[None],2,axis=0); defect,mse=mod.mapped_defect(L,constant.numpy(),exact.numpy(),duration.numpy()*.01)
    assert np.allclose(defect,0.,atol=1e-14,rtol=0) and np.allclose(mse,0.,atol=1e-14,rtol=0)
    ramp=torch.zeros((2,11,30),dtype=torch.float64); ramp[0,:2,0]=torch.tensor([0.,2.]); ramp[1,:4,0]=torch.tensor([0.,2.,4.,6.])
    short=mod.integrate_q(Drive(physical=True),ramp,torch.tensor([1,3]),torch.zeros((2,3),dtype=torch.float64),.005)
    assert torch.allclose(short[0],torch.full((3,),.01,dtype=torch.float64),atol=1e-14,rtol=0)
    assert torch.allclose(short[1],torch.full((3,),.09,dtype=torch.float64),atol=1e-14,rtol=0)


def test_windows_keep_terminal_short_interval_and_no_within_window_reset():
    assert mod.windows(31)==[(10,20),(20,30)]
    assert mod.windows(35)==[(10,20),(20,30),(30,34)]
    segment=torch.zeros((1,11,30),dtype=torch.float64)
    out=mod.integrate_q(Drive(),segment,torch.tensor([10]),torch.ones((1,3),dtype=torch.float64),.005)
    assert torch.allclose(out,torch.full((1,3),1.2,dtype=torch.float64),atol=1e-14,rtol=0)
    with pytest.raises(ValueError): mod.windows(11)


def test_nonzero_affine_q_dynamics_both_steps_and_exact_reference_defect():
    segment=torch.zeros((2,11,30),dtype=torch.float64); duration=torch.tensor([10,3]); q0=torch.tensor([[0.,1.,2.],[-1.,3.,.5]],dtype=torch.float64)
    time=duration.to(torch.float64)[:,None]*.01
    expected=q0*torch.exp(-2*time)+1.5*(1-torch.exp(-2*time))
    fine=mod.integrate_q(AffineQ(),segment,duration,q0,.005); coarse=mod.integrate_q(AffineQ(),segment,duration,q0,.01)
    assert torch.max(torch.abs(fine-expected)) < 1e-8
    assert torch.max(torch.abs(coarse-expected)) < 1e-8


def test_mapped_force_defect_is_coordinate_similarity_invariant():
    rng=np.random.default_rng(8); L=rng.normal(size=(4,10,3)); qhat=rng.normal(size=(4,3)); qref=rng.normal(size=(4,3)); duration=np.array([.1,.08,.03,.1])
    A=np.array([[1,.2,0],[0,2,.3],[.1,0,.5]])
    original,_=mod.mapped_defect(L,qhat,qref,duration)
    transformed,_=mod.mapped_defect(np.einsum('nij,jk->nik',L,np.linalg.inv(A)),qhat@A.T,qref@A.T,duration)
    assert np.allclose(original,transformed,atol=1e-12,rtol=0)


def test_event_weighting_and_nonfinite_fail_closed():
    event=np.r_[np.zeros(100,dtype=int),np.arange(1,346)]; labels=np.array(['stable']*311+['unstable']*35); groups=np.array(['1']*len(event)); values=np.r_[np.full(100,9.),np.ones(345)]
    report=mod.summarize(event,labels[event],groups,values)
    assert report['overall']['rms']==pytest.approx(np.sqrt((9.+345.)/346.))
    json.dumps(report, allow_nan=False)
    with pytest.raises(FloatingPointError): mod.integrate_q(Drive(),torch.full((1,11,30),float('nan')),torch.tensor([1]),torch.zeros((1,3),dtype=torch.float64),.01)
    with pytest.raises(FloatingPointError):
        mod._rhs(Drive(),torch.zeros((1,11,30),dtype=torch.float64),torch.tensor([1]),torch.full((1,3),float('nan'),dtype=torch.float64),0.)
