"""Synthetic-only fixed-form regression tests for Loop0091."""
import importlib.util, sys
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]; spec=importlib.util.spec_from_file_location('affine0091',ROOT/'experiments'/'audit_affine_signature_0091.py'); mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)

def fixture():
    rng=np.random.default_rng(91); one=np.ones(10); P=np.eye(10)-np.outer(one,one)/10
    U=rng.normal(size=(10,10))@P; V=rng.normal(size=(10,10)); omega=np.linspace(.4,2.2,13)
    Z=U[None]+1j*omega[:,None,None]*V[None]
    return Z,omega,U,V

def test_known_real_uv_retains_common_speed_and_right_only_u_gauge():
    Z,w,U,V=fixture(); u,v,pred,qa=mod.fit(Z,w,np.ones(13,bool))
    assert np.allclose(u,U,atol=1e-13,rtol=0) and np.allclose(v,V,atol=1e-13,rtol=0)
    assert np.linalg.norm(v@np.ones(10))>1e-6  # common input speed is retained
    assert np.max(abs(u@np.ones(10)))<1e-12 and np.linalg.norm(np.ones(10)@u)>1e-6
    assert qa['normal_u']<1e-12 and qa['normal_v']<1e-12 and np.allclose(pred,Z)

def test_nonrepresentable_component_is_not_clipped_from_capture():
    Z,w,_,_=fixture(); Z=Z+(w*w)[:,None,None]*np.eye(10)[None]
    u,v,pred,_=mod.fit(Z,w,np.ones(13,bool)); capture,_,_=mod._capture(Z,pred)
    assert capture<1 and np.isfinite(capture)

def test_loo_fit_has_exact_index_isolation():
    Z,w,_,_=fixture(); mask=np.arange(13)!=4; u,v,_,_=mod.fit(Z,w,mask)
    changed=Z.copy();changed[4]+=1000*(1+1j); u2,v2,_,_=mod.fit(changed,w,mask)
    assert not mask[4] and np.allclose(u,u2) and np.allclose(v,v2)

def test_bad_and_zero_inputs_fail_closed():
    Z,w,_,_=fixture()
    with pytest.raises(ValueError):mod.fit(Z,np.r_[w[:-1],np.nan],np.ones(13,bool))
    with pytest.raises(ArithmeticError):mod._capture(np.zeros_like(Z),np.zeros_like(Z))
