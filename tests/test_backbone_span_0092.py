"""Synthetic matrix checks for the fixed Loop0092 backbone span."""
import copy,importlib.util,sys,time
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('span0092',ROOT/'experiments'/'audit_backbone_span_0092.py');mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
sys.path.insert(0,str(ROOT/'experiments'))

def fixture():
    rng=np.random.default_rng(92);omega=np.linspace(.3,2.,13);mass=np.arange(1.,11.);D=rng.normal(size=(26,10));D-=D.mean(1,keepdims=True);phi=rng.normal(size=26);C=mod.basis(omega,mass,D,phi);beta=rng.normal(size=36);return omega,mass,D,phi,C,beta

def test_known_coefficients_and_common_speed_are_recovered():
    omega,mass,D,phi,C,beta=fixture();Z=C@beta;X,y=mod.stack(C,Z);got,pred,res,qa=mod.solve_span(X,y,time.perf_counter()+5,{'svd_attempts':0})
    assert np.allclose(got,beta,atol=1e-11,rtol=0) and np.linalg.norm(pred-y)<1e-10 and qa['rank']==36
    assert np.linalg.norm(C[:, :, :, 26:]@beta[26:])>0  # damping/common speed columns retained

def test_stiffness_and_damping_columns_have_registered_negative_signs():
    omega,mass,D,phi,C,_=fixture();L=mod.S_DELTA/(mass*mod.S_OMEGA)
    assert np.allclose(C[0,:,:,0],-np.cos(phi[0])*L[:,None]*np.outer(D[0],D[0]))
    assert np.allclose(C[3,:,:,30],-1j*omega[3]*L[:,None]*np.diag(np.eye(10)[4]))

def test_nonrepresentable_error_is_not_clipped():
    omega,mass,D,phi,C,beta=fixture();Z=C@beta+np.eye(10)[None]*(omega**2)[:,None,None];X,y=mod.stack(C,Z);_,_,_,qa=mod.solve_span(X,y,time.perf_counter()+5,{'svd_attempts':0})
    assert qa['capture']<1

def test_missing_nonfinite_and_rank_fail_closed():
    omega,mass,D,phi,C,beta=fixture()
    with pytest.raises(ValueError):mod.basis(omega,mass,D[:-1],phi)
    X,y=mod.stack(C,C@beta);X[:,1]=X[:,0]
    attempts={'svd_attempts':0}
    with pytest.raises(mod.SpanFailure) as failure:mod.solve_span(X,y,time.perf_counter()+5,attempts)
    assert attempts['svd_attempts']==1 and failure.value.singular.shape==(36,)

def canonical_payload():
    rng=np.random.default_rng(923);D=rng.normal(size=(26,10));D-=D.mean(1,keepdims=True);mass=np.arange(1.,11.);d=np.linspace(.2,.5,10)
    J=np.zeros((33,33));J[:10,10:20]=np.eye(10);J[10:20,:10]=-np.eye(10);R=np.zeros((33,33));R[10:20,10:20]=np.diag(d)
    return {'mass':mass.tolist(),'D':D.tolist(),'delta_star':np.linspace(-.01,.01,10).tolist(),'coupling':np.linspace(.3,.8,26).tolist(),'damping':d.tolist(),'gain':np.ones(10).tolist(),'n_p':10,'n_q':3,'J':J.tolist(),'R':R.tolist(),'B':np.zeros((33,0)).tolist()}

def test_nonlinear_export_copy_and_nonfinite_missing_fail_closed(tmp_path):
    payload=canonical_payload();D,mass,k,d,phi=mod.export_qa(payload);attempts={'rhs_evaluations':0}
    states,base,changed,difference,expected,_=mod.nonlinear_check(payload,D,mass,k,d,phi,attempts)
    assert states.shape==(3,33) and attempts['rhs_evaluations']==6 and np.allclose(difference,expected,atol=1e-13,rtol=0)
    bad=copy.deepcopy(payload);bad['mass'][0]=float('nan')
    with pytest.raises(ArithmeticError):mod.export_qa(bad)
    with pytest.raises(FileNotFoundError):mod.sha(tmp_path/'missing.json')
