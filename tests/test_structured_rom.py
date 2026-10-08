import sys
from pathlib import Path
import numpy as np
import torch
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))
from structure_grammar import PrimitiveSpec, primitive_value_and_gradient
from structured_rom import StructuredROM, numpy_field, AnchoredGRUEncoder


def model():
    return StructuredROM(mass=[2., 3.], D=[[1., -1.]], coupling=[1.2],
        delta_star=[.2, -.1], damping=[.1, .2], gain=[2., 3.], n_q=1,
        storage=[PrimitiveSpec("sin", (0,), -.3, phi=.3),
                 PrimitiveSpec("cos", (0,), .2, phi=.3),
                 PrimitiveSpec("quadratic", (1,), .1, (1.7,)),
                 PrimitiveSpec("product", (0, 2), -.2, (2., 1.)),
                 PrimitiveSpec("state_sin", (0, 0), .15, (2.,), .3),
                 PrimitiveSpec("state_cos", (2, 0), -.12, (1.,), .3)],
        exchange=[{"a": (1.,0.,0.,0.,0.), "b": (0.,0.,1.,0.,0.), "coefficient": -.3}],
        dissipation=[{"a": (1.,1.,0.,0.,0.), "coefficient": .2}],
        ports=[{"direction": (1.,0.,0.,0.,0.), "terms": [{"kind":"linear","index":0,"coefficient":.1},{"kind":"sin","index":0,"coefficient":.2}]}])


def test_all_primitive_gradients_match_autograd_and_fd():
    eta=torch.tensor([[.17]],dtype=torch.float64,requires_grad=True); s=torch.tensor([[.2,-.4,.7]],dtype=torch.float64,requires_grad=True)
    for spec in model().storage_specs:
        v,ge,gs=primitive_value_and_gradient(spec,eta,s); ae,ass=torch.autograd.grad(v.sum(),(eta,s),retain_graph=True,allow_unused=True)
        ae=torch.zeros_like(eta) if ae is None else ae; ass=torch.zeros_like(s) if ass is None else ass
        assert torch.allclose(ge,ae,atol=1e-12) and torch.allclose(gs,ass,atol=1e-12)


def test_model_gradient_power_and_gauge():
    m=model(); x=torch.randn(5,m.dim,dtype=torch.float64,requires_grad=True); e=m.gradient(x); auto=torch.autograd.grad(m.energy(x).sum(),x)[0]
    assert (e-auto).abs().max()<1e-10
    f,e,u,J,R=m.field(x,return_parts=True)
    assert torch.allclose((e*f).sum(-1),-(e@(R)).mul(e).sum(-1)+(e@m.port_matrix()).mul(u).sum(-1),atol=1e-10)
    shifted=x.detach().clone(); shifted[:,:m.n]+=1.9
    assert torch.allclose(m.field(x.detach()),m.field(shifted),atol=1e-11)
    assert torch.allclose(m.field(torch.zeros_like(x.detach())),torch.zeros_like(x.detach()),atol=1e-12)


def test_no_grad_export_and_trainable_families(tmp_path):
    m=model(); x=torch.randn(3,m.dim,dtype=torch.float64)
    with torch.no_grad(): f=m.field(x)
    loss=m.field(x).square().mean(); loss.backward()
    for family in (m.storage_gamma,m.exchange_kappa,m.dissipation_rho,m.port_direction,m.port_coefficient):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in family)
    torch.optim.SGD(m.parameters(),lr=1e-3).step()
    payload=m.export(tmp_path/'model.json'); expected=m.field(x).detach().numpy()
    assert np.max(np.abs(numpy_field(payload,x.numpy())-expected))<1e-10
    reloaded=StructuredROM.from_export(tmp_path/'model.json')
    assert torch.allclose(reloaded.field(x),m.field(x),atol=1e-10)


def test_exact_masks_paired_exchange_and_encoder():
    m=StructuredROM(mass=[1.,1.],D=[[1.,-1.]],coupling=[1.],gain=[2.,2.],jpi_p=[0.,0.],rpp=[0.,0.],n_q=0)
    J,R=m.matrices(); assert torch.count_nonzero(J[2:4,4:])==0 and torch.count_nonzero(R[4:,4:])==0
    e=model(); J,_=e.matrices(); assert torch.count_nonzero(J-J.T)>0
    enc=AnchoredGRUEncoder(); assert torch.equal(enc(torch.zeros(2,11,20,dtype=torch.float64)),torch.zeros(2,3,dtype=torch.float64))


def test_invalid_contracts_and_exact_zero_extra_dissipation(tmp_path):
    with pytest.raises(ValueError): StructuredROM(mass=[1.,1.],D=[[1.,0.]])
    with pytest.raises(ValueError): StructuredROM(mass=[1.,1.],D=[[1.,-1.]],gain=[1.,-1.])
    with pytest.raises(ValueError): StructuredROM(mass=[1.,1.],D=[[1.,-1.]],storage=[PrimitiveSpec('sq',(0,),scale=(0.,))])
    with pytest.raises(ValueError): StructuredROM(mass=[1.,1.],D=[[1.,-1.]],ports=[{'direction':(float('nan'),0.,0.,0.),'terms':[]}])
    m=StructuredROM(mass=[1.,1.],D=[[1.,-1.]],dissipation=[{'a':(1.,0.,0.,0.),'coefficient':0.}])
    _,r=m.matrices(); assert torch.count_nonzero(r)==0
    counts=m.export(tmp_path/'counts.json')['counts']
    assert counts['effective_parameter_coordinate_upper_bound'] <= counts['raw_trainable_scalar_count']
