"""Synthetic contracts for Loop0084; no learned model or private data are read."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import time
import json

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
spec=importlib.util.spec_from_file_location('null0084',ROOT/'experiments'/'audit_coefficient_nullspace_0084.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def analyze(D, fields=None):
    if fields is None: fields=np.zeros((2,33,24))
    return mod.analyze(D,{1e-5:D.copy(),1e-6:D.copy()},np.zeros(24),np.zeros((33,33)),fields)


def full_rank():
    D=np.zeros((33,24,33))
    for col in range(24): D[col,col,0]=1.
    return D


def test_known_full_rank_tensor_has_no_null_direction_and_cached_svd_reconstructs():
    D=full_rank(); D[7,3,11]=.25; summary,arrays=analyze(D)
    assert summary['numerical_rank']==24 and summary['nullity']==0
    assert summary['classification_flags']['all_resolved']
    assert arrays['normalized_D'][7*33+11,3] != 0  # [row,input,coefficient] stack order
    independent=np.linalg.svd(arrays['normalized_D'],full_matrices=False)[1]
    assert np.max(abs(independent-np.asarray(summary['singular_values'])))<1e-10


def test_exact_null_direction_can_have_quadratic_nonzero_fixed_state_field():
    D=full_rank();D[:,1,:]=0.  # coefficient one is exact Jacobian-null
    fields=np.zeros((2,33,24));fields[1,7,1]=.2
    summary,arrays=analyze(D,fields)
    assert 1 in summary['zero_columns'] and summary['classification_flags']['numerical_null']
    assert summary['classification_flags']['null_direction'] and summary['classification_flags']['nonlinear_detected']
    assert arrays['null_directions'].shape[0]>=1
    assert summary['nullity']==1
    json.dumps(summary,allow_nan=False)


def test_intermediate_singular_ratio_is_reported_ambiguous():
    D=full_rank();D[:,1,:]=0.;D[0,1,0]=1.;D[1,1,0]=1e-10
    summary,_=analyze(D)
    assert summary['classification_flags']['ambiguous']
    ratio=np.asarray(summary['singular_ratios'])[-1]
    assert 1e-12<ratio<1e-8


def test_actual_ad_fd_tensor_and_quadratic_null_field():
    D=full_rank(); D[:,1,:]=0.
    tensor=torch.tensor(D,dtype=torch.float64)
    quadratic=torch.zeros((33,24),dtype=torch.float64); quadratic[20,1]=1.
    def field(z): return torch.einsum('rcs,s->rc',tensor,z)+quadratic*z[20]**2
    zero=torch.zeros(33,dtype=torch.float64)
    ad=torch.autograd.functional.jacobian(field,zero).numpy()
    for h in mod.FD_STEPS:
        fd=mod.column_fd(lambda x:field(torch.tensor(x)).numpy(),h,time.perf_counter()+3)
        assert np.max(np.abs(fd-ad))<1e-12
    assert np.max(np.abs(ad-D))==0
    z=zero.clone(); z[20]=.2
    summary,_=analyze(ad,np.stack((field(zero).numpy(),field(z).numpy())))
    assert summary['nullity']==1 and summary['classification_flags']['nonlinear_detected']
