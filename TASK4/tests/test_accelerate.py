import copy
import torch
import pytest
from task4.synthetic import fixture
from task4.dynamics import PowerGraph,UnstructuredGraph
from task4.accelerate import trace_field

@pytest.mark.parametrize('cls',[PowerGraph,UnstructuredGraph])
def test_trace_preserves_values_gradients_parameter_updates_and_state(cls):
    _,phys=fixture();torch.manual_seed(501)
    m=cls(phys,[.2,.3,.1],hidden=8);j=copy.deepcopy(m);trace_field(j)
    assert set(m.state_dict())==set(j.state_dict())
    for batch in [1,2,5]:
        m.zero_grad();j.zero_grad()
        x=(torch.randn(batch,m.dim,dtype=torch.float64)*.1).requires_grad_()
        y=x.detach().clone().requires_grad_()
        f=m.field(x);g=j.field(y)
        torch.testing.assert_close(f,g,atol=1e-12,rtol=1e-10)
        f.square().sum().backward();g.square().sum().backward()
        torch.testing.assert_close(x.grad,y.grad,atol=1e-11,rtol=1e-9)
        for a,b in zip(m.parameters(),j.parameters()):
            if a.grad is not None:
                torch.testing.assert_close(a.grad,b.grad,atol=1e-10,rtol=1e-8)
                with torch.no_grad():a.add_(a.grad,alpha=-1e-4);b.add_(b.grad,alpha=-1e-4)
        torch.testing.assert_close(m.field(x,False),j.field(x,False),atol=1e-12,rtol=1e-10)

def test_traced_field_through_rk4_without_recomputation():
    from task4.integrators import rollout
    _,phys=fixture();torch.manual_seed(502)
    m=PowerGraph(phys,[.2,.3,.1],hidden=8);j=copy.deepcopy(m);trace_field(j)
    m.checkpoint_steps=j.checkpoint_steps=0
    x=(torch.randn(2,m.dim,dtype=torch.float64)*.1).requires_grad_();y=x.detach().clone().requires_grad_()
    a=rollout(m,x,3);b=rollout(j,y,3)
    torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-10)
    a.square().sum().backward();b.square().sum().backward()
    torch.testing.assert_close(x.grad,y.grad,atol=1e-10,rtol=1e-8)
    for p,q in zip(m.parameters(),j.parameters()):
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=1e-10,rtol=1e-8)
    j.checkpoint_steps=2
    with pytest.raises(ValueError,match='incompatible'):
        rollout(j,y,3)
