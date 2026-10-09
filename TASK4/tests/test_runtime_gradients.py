import copy
import numpy as np
import torch
from task4.synthetic import fixture
from task4.dynamics import PowerGraph
from task4.integrators import rollout

def test_analytic_chain_rule_and_parameter_gradient():
    _,phys=fixture();torch.manual_seed(94)
    model=PowerGraph(phys,[.2,.3,.1],hidden=8)
    x=torch.randn(2,model.dim,dtype=torch.float64,requires_grad=True)*.15
    old=torch.autograd.grad(model.energy(x).sum(),x,create_graph=True)[0]
    new=model.energy_gradient(x)
    torch.testing.assert_close(new,old,atol=1e-12,rtol=1e-10)
    pars=[p for name,p in model.named_parameters() if 'energy' in name]
    go=torch.autograd.grad(old.square().sum(),pars,allow_unused=True,retain_graph=True)
    gn=torch.autograd.grad(new.square().sum(),pars,allow_unused=True)
    for p,a,b in zip(pars,go,gn):
        torch.testing.assert_close(torch.zeros_like(p) if a is None else a,torch.zeros_like(p) if b is None else b,atol=1e-11,rtol=1e-9)

def test_recomputation_preserves_full_gradient():
    _,phys=fixture();torch.manual_seed(95)
    model=PowerGraph(phys,[.2,.3,.1],hidden=8);other=copy.deepcopy(model);other.checkpoint_steps=2
    x=(torch.randn(1,model.dim,dtype=torch.float64)*.1).requires_grad_(True)
    y=x.detach().clone().requires_grad_(True)
    a=rollout(model,x,4);b=rollout(other,y,4)
    torch.testing.assert_close(a,b,atol=1e-13,rtol=1e-12)
    a.square().sum().backward();b.square().sum().backward()
    torch.testing.assert_close(x.grad,y.grad,atol=1e-11,rtol=1e-9)
    for p,q in zip(model.parameters(),other.parameters()):
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=1e-11,rtol=1e-9)

def test_symbolic_support_survives_truth_finetuning():
    from task4.symbolic import PolynomialFunction
    m=PolynomialFunction([[0,0],[1,0],[0,1]],[[1.,0.],[0.,2.],[0.,0.]])
    opt=torch.optim.Adam(m.parameters(),lr=.01)
    x=torch.randn(20,2,dtype=torch.float64)
    m(x).square().sum().backward();opt.step()
    assert torch.count_nonzero(m.coefficients.detach()[~m.support])==0
    other=PolynomialFunction([[0,0],[1,0],[0,1]],[[1.,0.],[0.,2.],[0.,0.]])
    other.load_state_dict(m.state_dict())
    assert other.export()['nonzero_per_output']==[1,1]
