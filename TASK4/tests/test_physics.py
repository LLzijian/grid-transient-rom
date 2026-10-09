import copy
import numpy as np
import torch
from task4.data import Physics
from task4.synthetic import fixture
from task4.dynamics import PowerGraph,UnstructuredGraph
from task4.encoder import HistoryEncoder

def setup(strength=.01):
    _,p=fixture();torch.manual_seed(11)
    return PowerGraph(p,[.2,.3,.1],hidden=8,strength=strength),p

def test_power_and_kinematics():
    m,p=setup();x=torch.randn(3,m.dim,dtype=torch.float64)*.15
    f,d=m.field(x,details=True)
    assert torch.max(torch.abs(d['Hdot']-d['expected_Hdot']))<1e-10
    assert d['dissipation'].min()>=0
    assert torch.allclose(f[:,:m.n],x[:,m.n:2*m.n]/m.mass,atol=1e-12)

def test_origin_and_gauge():
    m,p=setup();x=torch.randn(2,m.dim,dtype=torch.float64)*.1
    x2=x.clone();x2[:,:m.n]+=2.3
    assert m.field(torch.zeros_like(x)).abs().max()<1e-12
    assert torch.allclose(m.field(x),m.field(x2),atol=1e-10,rtol=1e-9)

def test_energy_gradient_finite_difference():
    m,p=setup();x=torch.randn(1,m.dim,dtype=torch.float64)*.1
    _,d=m.field(x,details=True);numeric=[];h=1e-6
    for k in range(m.dim):
        a=x.clone();b=x.clone();a[:,k]+=h;b[:,k]-=h
        numeric.append(((m.energy(a)-m.energy(b))/(2*h)).item())
    assert np.max(np.abs(np.array(numeric)-d['gradient'].detach().numpy()[0]))<1e-7

def test_m3_embedding():
    m,p=setup(0.);x=torch.randn(2,m.dim,dtype=torch.float64)*.1
    z=m.physical(x);f=m.field(x);d=z[...,0].detach().numpy();w=z[...,1].detach().numpy();pm=z[...,2].detach().numpy()
    electric=((np.sin((d+p.delta_star)@p.incidence)-np.sin(p.delta_star@p.incidence))*p.coupling)@p.incidence.T
    np.testing.assert_allclose(f[:,m.n:2*m.n].detach().numpy(),pm-electric-p.damping*w,atol=1e-12)
    np.testing.assert_allclose(f[:,2*m.n:3*m.n].detach().numpy(),-p.rate*pm-p.gain*w,atol=1e-12)

def test_encoder_ignores_p_and_is_gauge_invariant():
    m,_=setup();e=HistoryEncoder(m,8)
    h=torch.randn(2,11,m.n,3,dtype=torch.float64)*.1;h2=h.clone();h2[...,2]+=100;h2[...,0]+=3.4
    assert torch.allclose(e(h),e(h2),atol=1e-10,rtol=1e-9)
    assert e(torch.zeros_like(h)).abs().max()==0

def test_machine_permutation():
    m,p=setup();perm=np.array([2,0,1])
    obj=p.as_dict()
    for key in ['mass','delta_star','pm_star','damping','rate','gain']:obj[key]=np.asarray(obj[key])[perm].tolist()
    obj['incidence']=p.incidence[perm].tolist();p2=Physics.from_dict(obj)
    m2=PowerGraph(p2,[.2,.3,.1],hidden=8)
    with torch.no_grad():
        for (name,a),(name2,b) in zip(m.named_parameters(),m2.named_parameters()):assert name==name2;b.copy_(a)
    x=torch.randn(2,m.dim,dtype=torch.float64)*.1
    d,pi,pm,q=m.split(x);x2=m2.assemble(d[:,perm],pi[:,perm],pm[:,perm],q[:,perm])
    a=m.field(x);ad,ap,am,aq=m.split(a)
    expected=m.assemble(ad[:,perm],ap[:,perm],am[:,perm],aq[:,perm])
    assert torch.allclose(expected,m2.field(x2),atol=1e-9,rtol=1e-8)

def test_edge_orientation():
    m,p=setup();obj=p.as_dict();obj['incidence']=(-p.incidence).tolist()
    m2=PowerGraph(Physics.from_dict(obj),[.2,.3,.1],hidden=8)
    with torch.no_grad():
        for (name,a),(name2,b) in zip(m.named_parameters(),m2.named_parameters()):assert name==name2;b.copy_(a)
    x=torch.randn(2,m.dim,dtype=torch.float64)*.1
    assert torch.allclose(m.field(x),m2.field(x),atol=1e-9,rtol=1e-8)

def test_gradient_reaches_encoder_and_ode():
    from task4.integrators import rollout
    m,_=setup();e=HistoryEncoder(m,8)
    h=torch.randn(2,11,m.n,3,dtype=torch.float64)*.1
    z=rollout(m,m.initial(h,e),2)
    m.physical(z[:,-1]).square().sum().backward()
    assert sum(float(p.grad.abs().sum()) for p in e.parameters() if p.grad is not None)>0
    assert sum(float(p.grad.abs().sum()) for p in m.node_energy.parameters() if p.grad is not None)>0

def test_u_control_no_power_claim():
    _,p=fixture();m=UnstructuredGraph(p,[1,1,1],hidden=8)
    f,d=m.field(torch.zeros(1,m.dim,dtype=torch.float64),details=True)
    assert f.abs().max()==0 and d['power_structure']=='not_applicable'
