"""Synthetic matrix contracts for Loop0083; no learned model is loaded."""
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pytest
torch=pytest.importorskip("torch")

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'experiments'))
spec=importlib.util.spec_from_file_location('poles0083',ROOT/'experiments'/'audit_coefficient_poles_0083.py')
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)


def matrix(unstable=False):
    A=np.zeros((33,33)); P=np.eye(10)-np.ones((10,10))/10
    A[:10,:10]=-2*P; A[10:,10:]=-3*np.eye(23)
    if unstable: A[20,20]=.25
    return A


def test_quotient_removes_only_common_angle_and_retains_uniform_speed():
    Q=mod.quotient(); common=np.r_[np.ones(10),np.zeros(23)]; speed=np.r_[np.zeros(10),np.ones(10),np.zeros(13)]
    assert Q.shape==(33,32)
    assert np.max(abs(Q.T@Q-np.eye(32)))<=1e-12
    assert np.max(abs(Q.T@common))<=1e-12
    assert np.linalg.norm(Q.T@speed)==pytest.approx(np.sqrt(10),abs=1e-14)


def test_linear_fd_and_quotient_spectra_recover_stable_and_unstable_cases():
    for unstable,sign in ((False,-1),(True,1)):
        A=matrix(unstable); fd={h:mod.central_fd(lambda x:A@x,h,time.perf_counter()+2) for h in mod.FD_STEPS}
        report,arrays=mod.analyze_matrix(A,fd)
        assert report['spectral_abscissa']*sign>1e-6
        assert report['qa']['eigen_residual_max']<1e-12
        assert all(value<1e-12 for value in report['qa']['fd_ad_relative'].values())
        assert report['qa']['q_common_angle_max']<=1e-12
        assert report['qa']['sensitivity_radius']<report['qa']['sensitivity_limit']
        assert np.allclose(arrays['A32'],arrays['Q'].T@A@arrays['Q'],atol=1e-14,rtol=0)


class Linear33(torch.nn.Module):
    def __init__(self, unstable=False):
        super().__init__(); A=matrix(unstable); P=np.eye(10)-np.ones((10,10))/10
        A[:10,:]=0.; A[:10,10:20]=np.eye(10)  # delta_dot = omega
        A[10:20,:10]=-2*P; A[10:20,10:20]=-3*np.eye(10)
        self.register_buffer("A",torch.as_tensor(A,dtype=torch.float64))
    def field(self,z): return z@self.A.T


def test_model_block_uses_ad_and_two_fd_for_actual_torch_linear_field():
    for unstable, sign in ((False,-1),(True,1)):
        report,arrays=mod._model_block("linear",Linear33(unstable),time.perf_counter()+5)
        assert report["spectral_abscissa"]*sign>1e-6
        assert all(report["qa"]["flags"].values())
        assert arrays["A33"].shape==(33,33)
        assert {"FD1e-05_A33","FD1e-06_A33"}.issubset(arrays)


def test_fd_deadline_fails_closed():
    with pytest.raises(TimeoutError): mod.central_fd(lambda x:x,1e-5,time.perf_counter()-1)


def test_saved_matrix_reconstruction_is_independent_of_field_calls():
    A=matrix(True); Q=mod.quotient(); saved=Q.T@A@Q
    reconstructed=Q.T@A@Q
    reference=max(1.,abs(np.max(np.linalg.eigvals(saved).real)))
    assert np.max(abs(saved-reconstructed))/reference<1e-10
    assert abs(np.max(np.linalg.eigvals(saved).real)-np.max(np.linalg.eigvals(reconstructed).real))/reference<1e-10
