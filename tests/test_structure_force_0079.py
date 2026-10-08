import importlib.util, sys
from pathlib import Path
import numpy as np, pytest
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'experiments'))
spec=importlib.util.spec_from_file_location('force0079',ROOT/'experiments/audit_structure_force_0079.py'); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

def test_affine_projection_in_out_span_and_similarity():
    rng=np.random.default_rng(1); L=rng.normal(size=(4,10,3)); q=rng.normal(size=(4,3)); inside=np.einsum('nij,nj->ni',L,rng.normal(size=(4,3))); outside=np.zeros((4,10)); outside[:,0]=1
    a=mod.project(inside,L,q); b=mod.project(outside,L,q)
    assert np.max(abs(a['e_perp']))<1e-11
    assert np.linalg.norm(b['e_perp'])>0
    block={'T':inside,'h0':np.zeros_like(inside),'L':L,'qenc':q,'h_qenc':np.einsum('nij,nj->ni',L,q),'h_fixed':np.einsum('nij,j->ni',L,mod.QFIX),**a}
    assert mod.qa(block)['passed']

def test_deficient_zero_rank_equal_event_and_nonfinite_fail_closed():
    L=np.zeros((2,10,3)); u=np.ones((2,10)); q=np.zeros((2,3)); x=mod.project(u,L,q)
    assert np.array_equal(x['ranks'],[0,0]) and np.allclose(x['e_perp'],u)
    rows=[{'event_id':'a','group':'1','label':'stable','e_perp':1.,'e_parallel':3.},{'event_id':'b','group':'1','label':'stable','e_perp':9.,'e_parallel':1.}]
    assert mod._aggregate(rows)['e_perp']==pytest.approx(5.)
    with pytest.raises(FloatingPointError): mod.project(np.array([[np.nan]*10]),np.zeros((1,10,3)),np.zeros((1,3)))

@pytest.mark.parametrize('rank',[1,2])
def test_rank_deficient_svd_projection(rank):
    rng=np.random.default_rng(rank); basis=rng.normal(size=(10,rank)); coeff=rng.normal(size=(3,rank)); L=np.repeat((basis@coeff.T)[None],2,axis=0)
    u=np.einsum('nij,nj->ni',L,rng.normal(size=(2,3))); out=mod.project(u,L,np.zeros((2,3)))
    assert np.array_equal(out['ranks'],np.full(2,rank))
    assert np.max(abs(out['e_perp']))<1e-10

def test_endpoint_block_target_and_event_balanced_bin_contract():
    from test_train_structured_rom import source_fixture
    import torch
    base,_=source_fixture()
    class Affine:
        mass=torch.as_tensor(base.mass,dtype=torch.float64)
        def encode(self,h): return h[:,-1,:3]
        def field(self,z):
            out=torch.zeros_like(z); out[:,10:20]=z[:,30:]@torch.ones((3,10),dtype=z.dtype); return out
    event={'x':np.zeros((14,30)), 'force':np.zeros((14,10)), 'id':'synthetic','group':'1','label':'stable'}
    block=mod.endpoint_block(Affine(),base,event,np.array([10,11,12]))
    assert block['T'].shape==(3,10) and block['L'].shape==(3,10,3)
    short=[{'event_id':'a','group':'1','label':'stable','n_endpoints':1,'raw_force_sq':1.,'e_perp':1.,'e_parallel':1.,'e_total':2.,'net_e_perp':1.,'net_e_parallel':1.,'net_e_total':2.,'common_accel_e_perp':1.,'common_accel_e_parallel':1.,'common_accel_e_total':2.},{'event_id':'b','group':'1','label':'stable','n_endpoints':100,'raw_force_sq':9.,'e_perp':9.,'e_parallel':9.,'e_total':18.,'net_e_perp':9.,'net_e_parallel':9.,'net_e_total':18.,'common_accel_e_perp':9.,'common_accel_e_parallel':9.,'common_accel_e_total':18.}]
    assert mod._report(short)['e_perp']==pytest.approx(5.)
    assert mod._report(short)['n_endpoints']==101

def test_true_force_reverses_only_frozen_backbone():
    from test_train_structured_rom import source_fixture
    base,_=source_fixture(); rng=np.random.default_rng(9)
    x=rng.normal(size=(13,30)); torque=rng.normal(size=(13,10))
    phase=base.delta_star@base.incidence
    e0=((np.sin(x[:,:10]@base.incidence+phase)-np.sin(phase))*base.coupling)@base.incidence.T
    event={'x':x,'force':torque-e0-x[:,10:20]*base.damping}
    np.testing.assert_allclose(mod.target(base,event,np.array([10,12])),torque[[10,12]],rtol=1e-12,atol=1e-12)

def test_qa_rejects_nonfinite_output():
    with pytest.raises(FloatingPointError): mod._relative(np.array([np.nan]),np.array([1.]))
