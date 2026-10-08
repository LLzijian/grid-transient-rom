import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"experiments"))
import backbone_compatibility_0095 as m

def test_known_basis_constant_and_factor_reconstruction():
    x=np.zeros((6,30)); D=np.zeros((26,10)); D[0,0]=1.; phi=np.zeros(26)
    B=m.known_basis(x,np.ones(10),D,phi,.005)
    assert B.shape==(6,10,36) and np.isfinite(B).all()
    G=np.tile(np.eye(37),(3,346,1,1)); F,e,r=m.factors(G); assert F.shape==G.shape and np.max(r)<=1e-10 and np.allclose(F[0,0].T@F[0,0],G[0,0])

def test_nonlinear_quadrature_distinguishes_registered_grids():
    x=np.zeros((3,30)); x[:,0]=[0.,.7,1.4]; D=np.zeros((26,10)); D[0,0]=1.; phi=np.array([.3]+[0.]*25)
    fine=m.known_basis(x,np.ones(10),D,phi,.005); coarse=m.known_basis(x,np.ones(10),D,phi,.01)
    assert not np.allclose(fine[1],coarse[1]) and np.isfinite(fine).all() and np.isfinite(coarse).all()
    values=-np.sin(.3+.7*np.array([0,.25,.5,.75,1]))+np.sin(.3)
    assert fine[1,0,0]==pytest.approx(.005/6*np.dot([1,4,2,4,1],values))
    assert coarse[1,0,0]==pytest.approx(.01/6*np.dot([1,4,1],values[[0,2,4]]))


def test_nonzero_constant_and_linear_integrals_have_exact_scale():
    x=np.zeros((4,30)); x[:,0]=.4; times=np.arange(4)*.01
    x[:,10:20]=(2+3*times)[:,None]
    mass=np.arange(1.,11.); D=np.zeros((26,10)); D[0,:2]=[1,-1]; phi=np.zeros(26)
    for dt in (.005,.01):
        B=m.known_basis(x,mass,D,phi,dt)
        np.testing.assert_allclose(B[:,0,0],-np.sin(.4)*times)
        np.testing.assert_allclose(B[:,1,0],np.sin(.4)*times/2)
        for j in range(10):
            np.testing.assert_allclose(B[:,j,26+j],-(2*times+1.5*times**2)/mass[j])


def test_nonzero_weighted_grams_and_required_increment_anchors(tmp_path):
    n=2; mass=np.arange(1.,11.); D=np.zeros((26,10)); D[0,:2]=[1,-1]
    phi=np.zeros(26); dk=np.r_[.7,np.zeros(25)]; dd=np.arange(10.)*.01
    truth=np.full((346,501,30),np.nan); truth[:,:3]=0.
    truth[:,:3,0]=[.1,.4,.7]; truth[:,:3,10:20]=2.
    fine=np.full((346,1001,33),np.nan); coarse=np.full((346,501,33),np.nan)
    fine[:,:5]=0.;coarse[:,:3]=0.
    for grid,dt in ((fine,.005),(coarse,.01)):
        B=m.known_basis(truth[0,:3],mass,D,phi,dt); idx=np.arange(3)*(2 if dt==.005 else 1)
        grid[:,idx,13:23]=B[:,:,:26]@dk; grid[:,idx,23:33]=B[:,:,26:]@dd
        grid[:,idx[1:],3:13]=m.SOMEGA
    data=dict(fine=fine,coarse=coarse,truth=truth,mass=mass,D=D,phi=phi,
              lengths=np.full(346,n),delta_k=dk,delta_d=dd)
    scales=np.linspace(.5,2.,36); out=m.build_grams(data,scales)
    np.testing.assert_allclose(out['grams'][...,36,36],1.)
    assert out['anchors']['increment_reconstruction_max_relative']<1e-14
    v=np.r_[dk,dd]/scales
    actual=m.score(out['grams'],v)
    for gi,(grid,idx) in enumerate(((fine,[2,4]),(coarse,[1,2]))):
        residual=grid[0,idx,3:13]+grid[0,idx,13:23]+grid[0,idx,23:33]
        wr=m._W(residual[:,:,None],mass)
        expected=np.mean(np.sum(wr[:,:,0]**2,axis=-1))
        np.testing.assert_allclose(actual[gi],expected)
    assert np.max(out['difference_grams'])>0
    data['fine'][0,2,13]+=1.
    with pytest.raises(AssertionError):
        m.build_grams(data,scales,partial_path=tmp_path/'partial.npz')
    with np.load(tmp_path/'partial.npz') as z: assert z['completed_events']==0


def test_factor_orientation_for_nondiagonal_gram():
    F0=np.eye(37); F0[0,1]=.3; F0[36,1]=-.2
    G=np.tile(F0.T@F0,(3,346,1,1)); counters={}
    F,e,r=m.factors(G,counters=counters)
    assert counters['eighs']==1038
    v=np.arange(37)/37
    assert np.sum((F[0,0]@v)**2)==pytest.approx(v@G[0,0]@v)

def test_score_and_risk_schema_with_negative_cross_terms():
    G=np.tile(np.eye(37),(2,3,346,1,1)); v=np.zeros(36); S=m.score(G,v); out=m.risk_summary(S,np.array(["stable"]*311+["unstable"]*35))
    assert S.shape==(2,3,346) and out["grids"]["fine"]["5"]["severe_worst4"]==pytest.approx(1.)

def test_basis_rejects_nonfinite():
    x=np.zeros((3,30)); x[1,0]=np.nan
    with pytest.raises(FloatingPointError): m.known_basis(x,np.ones(10),np.zeros((26,10)),np.zeros(26),.005)
