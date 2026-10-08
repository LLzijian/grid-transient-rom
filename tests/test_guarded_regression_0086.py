"""Synthetic KKT and tensor-order contracts for Loop0086; no learned artifact is opened."""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"experiments"))
spec=importlib.util.spec_from_file_location("guard0086",ROOT/"experiments"/"fit_guarded_regression_0086.py")
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def _lmis(): return np.zeros((24,32,32))


def test_unconstrained_feasible_sdp_has_zero_dual_and_kkt():
    K=np.eye(24);g=np.linspace(-.1,.1,24);out=mod.solve_sdp(K,g,_lmis())
    assert np.allclose(out["beta"],g,atol=1e-8)
    assert out["F_min_eigenvalue"]==pytest.approx(.49)
    assert out["stationarity"]<1e-7 and out["gap"]<1e-7


def test_binding_diagonal_lmi_has_feasible_dual_kkt_certificate():
    L=_lmis();L[0]=np.eye(32);g=np.zeros(24);g[0]=1.
    out=mod.solve_sdp(np.eye(24),g,L)
    assert out["beta"][0]==pytest.approx(.49,abs=2e-6)
    assert out["F_min_eigenvalue"]>=-1e-8 and out["Y_min_eigenvalue"]>=-1e-8
    assert max(out["stationarity"],out["complementarity"],out["gap"])<1e-6
    assert np.allclose(out["Y"],out["Y"].T) and np.isfinite(out["Y_raw"]).all()
    json.dumps(mod.jsonable(out),allow_nan=False)


def test_rank_one_offdiagonal_lmi_has_registered_dual_scaling():
    v=np.zeros(32);v[:2]=1/np.sqrt(2);L=np.einsum("i,j->ij",v,v)[None]
    out=mod.solve_sdp(np.ones((1,1)),np.ones(1),L)
    assert out["beta"][0]==pytest.approx(.49,abs=2e-6)
    assert float(np.einsum("ab,ab",out["Y"],L[0]))==pytest.approx(1.02,abs=3e-6)
    assert out["F_min_eigenvalue"]>=-1e-8 and out["complementarity"]<1e-6


def test_tensor_coefficient_order_is_row_coefficient_input():
    Q=np.r_[np.eye(32),np.zeros((1,32))];P=.5*np.eye(32);D=np.zeros((33,24,33));D[3,7,4]=6.;scales=np.ones(24);scales[7]=3.
    L=mod.coefficient_lmis(D,Q,scales,P)
    assert L.shape==(24,32,32)
    assert L[7,3,4]==pytest.approx(1.) and L[7,4,3]==pytest.approx(1.)
    assert not np.any(L[np.arange(24)!=7])


def test_solver_and_tensor_reject_invalid_shapes_and_nonfinite_inputs():
    with pytest.raises(ValueError):mod.solve_sdp(np.eye(23),np.zeros(24),_lmis())
    with pytest.raises(ValueError):mod.solve_sdp(np.eye(24),np.zeros((24,1)),_lmis())
    with pytest.raises(ValueError):mod.kkt(np.eye(1),np.ones(1),np.zeros((1,32,32)),np.ones(1),np.zeros(32))
    with pytest.raises(ValueError):mod.coefficient_lmis(np.zeros((33,24,33)),np.eye(33),np.ones(24),np.eye(32))
    D=np.zeros((33,24,33));D[0,0,0]=np.nan
    with pytest.raises(ValueError):mod.coefficient_lmis(D,np.r_[np.eye(32),np.zeros((1,32))],np.ones(24),np.eye(32))


def test_conditional_preserves_nonfinite_raw_arrays_for_failure_artifact():
    design={"c":np.full((1,31),np.nan),"A":np.zeros((1,31,24)),"coarse_c":np.zeros((1,31)),"coarse_A":np.zeros((1,31,24))}
    summary,arrays=mod._conditional({},design,np.zeros(24))
    assert summary["finite"] is False and np.isnan(arrays["fine_candidate"]).all()
