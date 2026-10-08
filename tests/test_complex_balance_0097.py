"""Synthetic-only complex directional-balance contracts for Loop0097."""
import importlib.util,sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'));spec=importlib.util.spec_from_file_location('balance0097',ROOT/'experiments'/'audit_complex_balance_0097.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)

def _stable_block():
    A=np.zeros((33,33));A[:10,10:20]=np.eye(10);A[20:,20:]=-np.eye(13);return A

def test_phase_amplitude_invariance_and_common_delta_retention():
    A=_stable_block();mass=np.ones(10);lam=1j;v=np.zeros(33,dtype=complex);v[:10]=1.
    one=m.complex_balance(A,mass,lam,v,float('inf'));two=m.complex_balance(A,mass,lam,(2-3j)*v,float('inf'))
    assert one['qa_pass'] and two['qa_pass'] and one['eta']==pytest.approx(two['eta'])
    assert np.linalg.norm(one['K']@(v[:10]/m.SDELTA))>0  # common delta was not projected away

def test_output_normalization_and_zero_denominator_fail_closed():
    mass=np.arange(1.,11.);v=np.zeros(699,dtype=complex);v[:10]=np.arange(10);v[10:20]=1.
    n,norm=m.normalize_target(v,mass);assert norm>0 and np.allclose(m.physical_output(n,mass),m.physical_output(v,mass)/norm)
    with pytest.raises(ValueError):m.normalize_target(np.zeros(699,complex),mass)
    with pytest.raises(ArithmeticError):m.complex_balance(_stable_block(),np.ones(10),1j,np.zeros(33,complex),float('inf'))

def test_wrong_units_and_singular_hidden_fail_closed():
    lam=1j;mass=np.ones(10);EF=np.ones(699);EF[10:20]=m.OMEGA
    AF=np.zeros((699,699),complex);AF[:10,10:20]=m.OMEGA*np.eye(10);AF[10:20,10:20]=lam*m.OMEGA*np.eye(10)
    A=_stable_block();v=np.zeros(699,complex);v[0]=1.;v[10]=lam/m.OMEGA
    assert m.controls(AF,EF,A,mass,lam,v)['fom_backward']<1e-12
    badomega=v.copy();badomega[10]=lam/(2*m.OMEGA)
    with pytest.raises(ArithmeticError):m.controls(AF,EF,A,mass,lam,badomega)
    tail=AF.copy();tail[0,25]=1.
    with pytest.raises(ArithmeticError):m.controls(tail,EF,A,mass,lam,v)
    back=AF.copy();back[300,0]=1.
    with pytest.raises(ArithmeticError):m.controls(back,EF,A,mass,lam,v)
    singular=np.zeros((33,33));singular[:10,10:20]=np.eye(10);singular[20:,20:]=-np.eye(13)
    got=m.complex_balance(singular,np.ones(10),-1.,np.r_[np.ones(10),np.zeros(23)],float('inf'))
    assert not got['qa_pass']

def test_nonzero_hidden_coupling_matches_closed_form_T_and_X():
    A=_stable_block();A[20,0]=2.;A[10,0]=4.;A[10,10]=5.;A[10,20]=3.;lam=1j
    got=m.complex_balance(A,np.ones(10),lam,np.r_[np.array([1.]),np.zeros(32)],float('inf'))
    assert got['qa_pass'] and got['X'][0,0]==pytest.approx(1-1j)
    assert got['T'][0,0]==pytest.approx(4+5j+3*(1-1j))
