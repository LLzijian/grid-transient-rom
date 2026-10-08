"""Synthetic-only contracts for the fixed conditioned 0094 replay."""
import importlib.util, sys
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('balance0094',ROOT/'experiments'/'audit_conditional_balance_0094.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)

def test_piecewise_linear_endpoint_and_no_extrapolation():
    truth=np.array([[[0.],[2.],[6.]]])
    assert m.physical_at(truth,0,.005)[0]==pytest.approx(1.)
    assert m.physical_at(truth,0,.02)[0]==pytest.approx(6.)
    with pytest.raises(ValueError):m.physical_at(truth,0,.025)

def test_known_terms_and_omega_only_field_difference():
    D=np.zeros((26,10));D[0,0]=1.;mass=np.ones(10);phi=np.zeros(26);x=np.zeros(30);x[0]=.2;x[10]=3.
    fk,fd=m.known_terms(x,mass,D,phi,np.r_[2.,np.zeros(25)],np.r_[4.,np.zeros(9)])
    assert fk[0]==pytest.approx(-2*np.sin(.2)) and fd[0]==pytest.approx(-12.)
    diff=m.expected_field_difference(x,mass,D,phi,np.r_[2.,np.zeros(25)],np.r_[4.,np.zeros(9)])
    assert np.count_nonzero(diff)==1 and diff[10]==pytest.approx(fk[0]+fd[0])
    assert m.probe_qa(np.zeros((3,33)),np.tile(diff,(3,1)),np.tile(diff,(3,1)))==pytest.approx(0.)
    with pytest.raises(FloatingPointError):m.probe_qa(np.full((3,33),np.nan),np.zeros((3,33)),np.zeros((3,33)))

def test_packed_rk4_linear_q_and_masked_nan_padding(tmp_path):
    import torch
    class Linear:
        def begin_graph_cache(self):pass
        def end_graph_cache(self):pass
        def field(self,x):
            y=torch.zeros_like(x);y[:,30:33]=-2*x[:,30:33]+3.;y[:,10:20]=3.;return y
    truth=np.zeros((346,3,30));qenc=np.tile(np.array([0.,1.,2.]),(346,1));lengths=np.ones(346,dtype=int)
    out=m.replay(Linear(),truth,qenc,lengths,np.ones(10),np.zeros((26,10)),np.zeros(26),np.zeros(26),np.zeros(10),.01,float('inf'),{},tmp_path/'partial.npz')
    expected=1.5+(qenc[0]-1.5)*np.exp(-.02)
    assert np.max(np.abs(out[0,1,:3]-expected))<1e-8
    assert np.allclose(out[0,1,3:13],.03,atol=1e-14) and np.isnan(out[0,2:]).all()

def test_replay_preserves_partial_on_forced_field_failure(tmp_path):
    class Broken:
        def begin_graph_cache(self):pass
        def end_graph_cache(self):pass
        def field(self,x):raise FloatingPointError('synthetic field failure')
    truth=np.zeros((346,2,30));q=np.zeros((346,3));partial=tmp_path/'partial.npz'
    with pytest.raises(FloatingPointError):m.replay(Broken(),truth,q,np.ones(346,int),np.ones(10),np.zeros((26,10)),np.zeros(26),np.zeros(26),np.zeros(10),.01,float('inf'),{},partial)
    with np.load(partial,allow_pickle=False) as z: assert z['packed'].shape==(346,501,33) and np.isfinite(z['packed'][:,0]).all()

def test_nonfinite_or_bad_prefix_fails_closed():
    with pytest.raises(FloatingPointError):m.known_terms(np.full(30,np.nan),np.ones(10),np.zeros((26,10)),np.zeros(26),np.zeros(26),np.zeros(10))
    with pytest.raises(ValueError):m.physical_at(np.zeros((1,1,30)),0,.01)

def test_cache_genuine_prefix_and_nan_padding_contract():
    truth=np.full((346,501,30),np.nan);truth[:,0]=0.;truth[:,1]=0.
    cache={'event_ids':np.arange(346),'labels':np.array(['stable']*311+['unstable']*35),'lengths':np.ones(346,dtype=int),'truth':truth,'qenc':np.zeros((346,3)),'mass':np.ones(10),'p_scale':np.ones(10)}
    assert np.array_equal(m.validate_cache(cache),np.ones(346,dtype=int))
    cache['truth'][0,2,0]=0.
    with pytest.raises(AssertionError):m.validate_cache(cache)
