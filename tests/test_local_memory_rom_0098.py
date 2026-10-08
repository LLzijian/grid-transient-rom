import sys
from pathlib import Path

import numpy as np
import torch
import pytest
import copy

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))
from local_memory_rom_0098 import LocalMemoryROM
from structured_rom import StructuredROM


def _base():
    D = torch.zeros((26, 10), dtype=torch.float64)
    D[:9] = torch.eye(10, dtype=torch.float64)[:9] - torch.eye(10, dtype=torch.float64)[1:]
    D[9] = torch.tensor([1., 0, 0, 0, 0, 0, 0, 0, 0, -1.])
    D[10:] = D[:16].clone()
    return StructuredROM(mass=torch.linspace(1., 2., 10), D=D, coupling=torch.ones(26),
                         delta_star=torch.linspace(-.1, .1, 10), damping=torch.full((10,), .1),
                         gain=torch.ones(10), n_q=0, storage=()).double()


def test_arms_and_shared_encoder_shapes():
    h = torch.randn(3, 11, 30, dtype=torch.float64)
    for arm, rank in (("G3-S", 3), ("L10-S", 10), ("HL11-S", 11)):
        model = LocalMemoryROM(_base(), torch.ones(20), torch.ones(10), arm)
        assert model.dim == 30 + rank
        assert model.encode(h).shape == (3, rank)
        assert model.W.shape == (10, rank)


def test_analytic_gradient_power_and_masks():
    model = LocalMemoryROM(_base(), torch.ones(20), torch.ones(10), "HL11-S")
    x = torch.randn(4, model.dim, dtype=torch.float64, requires_grad=True)
    H, e = model.energy_and_gradient(x)
    auto = torch.autograd.grad(H.sum(), x)[0]
    assert torch.allclose(e, auto, atol=1e-11)
    J, R = model.matrices()
    assert torch.allclose(J + J.T, torch.zeros_like(J), atol=1e-12)
    assert torch.linalg.eigvalsh(R).min() >= -1e-12
    f = model.internal_field(x)
    u = model.port_values(x)
    B = model.port_matrix()
    balance = (e * f).sum(-1) + (e @ R * e).sum(-1) - (u * (e @ B)).sum(-1)
    assert balance.abs().max() < 1e-10
    model.set_active(0, True)
    assert model.template_coefficients[0] != 0
    model.set_active(0, False)
    assert model.template_coefficients[0] == 0


def test_encoder_does_not_read_p_or_future_and_reload(tmp_path):
    model = LocalMemoryROM(_base(), torch.ones(20), torch.ones(10), "L10-S")
    h = torch.randn(2, 11, 30, dtype=torch.float64)
    q = model.encode(h)
    changed = h.clone()
    changed[..., 20:30] += 100.
    assert torch.equal(q, model.encode(changed))
    model.set_active(7, True)
    path = tmp_path / "local_memory.json"
    model.export(path)
    restored = LocalMemoryROM.from_export(path)
    z = torch.randn(2, model.dim, dtype=torch.float64)
    np.testing.assert_allclose(model.field(z).detach().numpy(), restored.field(z).detach().numpy(), atol=1e-11)


def test_complexity_penalizes_port_coefficients_not_a_discrete_port_count():
    model=LocalMemoryROM(_base(),torch.ones(20),torch.ones(10),'L10-S')
    model.set_active(7,True)
    penalty=model.complexity()
    port_grad=torch.autograd.grad(penalty,model.port_raw)[0]
    assert torch.equal(port_grad,torch.ones_like(port_grad))
    assert torch.equal(penalty,model.complexity(torch.randn(2,model.dim,dtype=torch.float64)))


def test_single_state_jacobian_zero_activation_gradients_and_m3_embedding():
    model = LocalMemoryROM(_base(), torch.ones(20), torch.ones(10), "L10-S")
    x = torch.randn(model.dim, dtype=torch.float64, requires_grad=True)
    H, e = model.energy_and_gradient(x)
    assert H.ndim == 0 and e.shape == x.shape
    jac = torch.autograd.functional.jacobian(model.internal_field, x)
    assert jac.shape == (model.dim, model.dim) and torch.isfinite(jac).all()
    for i in range(8):
        model.set_active(i, True)
        for p in model.template_parameters(i):
            with torch.no_grad():
                p.zero_()
        loss = model.internal_field(x).square().mean()
        grads = torch.autograd.grad(loss, model.template_parameters(i), allow_unused=True)
        assert any(g is not None and torch.isfinite(g).all() and torch.any(g.abs() > 0) for g in grads)
        model.set_active(i, False)

    # With all optional templates masked and q=0, the mechanical block is the
    # copied M3 StructuredROM field exactly.
    for i in range(8):
        model.set_active(i, False)
    z = torch.randn(2, model.dim, dtype=torch.float64)
    base_z = z[..., :30]
    base_with_q = torch.cat((base_z, torch.zeros(2, 10, dtype=torch.float64)), -1)
    expected = model.structured.interface_field(base_with_q)
    actual = model.field(base_with_q)[..., :30]
    assert torch.allclose(actual, expected[..., :30], atol=1e-11)


def test_edge_orientation_and_machine_permutation_equivariance():
    base = _base()
    model = LocalMemoryROM(base, torch.ones(20), torch.ones(10), "HL11-S")
    for i in (0, 1, 2, 3, 5, 6, 7):
        model.set_active(i, True)
    z = torch.randn(model.dim, dtype=torch.float64)
    # Reversing one incidence row leaves both the oriented sine and endpoint
    # cosine terms unchanged.
    flipped = _base()
    flipped.D[0] *= -1
    flipped_model = LocalMemoryROM(flipped, torch.ones(20), torch.ones(10), "HL11-S")
    for i in (0, 1, 2, 3, 5, 6, 7):
        flipped_model.set_active(i, True)
    assert torch.allclose(model.field(z), flipped_model.field(z), atol=1e-10)

    # Reordering machines, physical parameters, and state only reorders local
    # q outputs and the corresponding field coordinates.
    perm = torch.tensor([3, 0, 7, 1, 9, 2, 8, 4, 6, 5])
    pbase = _base()
    pbase.D = pbase.D[:, perm]
    pbase.mass = pbase.mass[perm]
    pbase.delta_star = pbase.delta_star[perm]
    pmodel = LocalMemoryROM(pbase, torch.ones(20), torch.ones(10), "HL11-S")
    for i in (0, 1, 2, 3, 5, 6, 7):
        pmodel.set_active(i, True)
    zp = torch.cat((z[:10][perm], z[10:20][perm], z[20:30][perm], z[30:40][:10][perm], z[40:]))
    out = model.field(z)
    pout = pmodel.field(zp)
    expected = torch.cat((out[:10][perm], out[10:20][perm], out[20:30][perm], out[30:40][:10][perm], out[40:]))
    assert torch.allclose(pout, expected, atol=1e-9)


@pytest.mark.parametrize('arm',('G3-S','L10-S','HL11-S'))
def test_heterogeneous_full_template_permutation_encoder_and_gradient(arm,tmp_path):
    from kiki.models.private_pm import MechanicalROM
    from kiki.physics import complete_incidence
    base=MechanicalROM(np.linspace(1.,2.,10),complete_incidence(10)[0],
        np.linspace(-.2,.2,10),np.linspace(.3,.6,10),np.r_[np.ones(26)*.2,np.zeros(19)],
        np.linspace(.03,.1,10),np.linspace(.1,.8,10),np.linspace(.2,.7,10))
    scales=np.linspace(.5,2.,20); ps=np.linspace(.3,1.,10)
    model=LocalMemoryROM(base,scales,ps,arm)
    perm=np.array([3,0,7,1,9,2,8,4,6,5])
    other=copy.deepcopy(base)
    for attr in ('mass','delta_star','pm_star','damping','rate','gain'):
        setattr(other,attr,getattr(base,attr)[perm])
    other.incidence=base.incidence[perm]
    pmodel=LocalMemoryROM(other,np.r_[scales[:10][perm],scales[10:][perm]],ps[perm],arm)
    for i in range(8):
        model.set_active(i,True); pmodel.set_active(i,True)
    coordinate=np.r_[perm,perm+10,perm+20,
                     np.arange(30,33) if arm=='G3-S' else (perm+30 if arm=='L10-S' else np.r_[perm+30,40])]
    h=torch.as_tensor(np.random.default_rng(2).normal(scale=.05,size=(3,11,30)),dtype=torch.float64)
    hp=h[...,np.r_[perm,perm+10,perm+20]]
    qp=coordinate[30:]-30
    assert torch.allclose(pmodel.encode(hp),model.encode(h)[...,qp],atol=2e-13,rtol=0)
    z=torch.randn(3,model.dim,dtype=torch.float64)*.1
    assert torch.allclose(pmodel.field(z[:,coordinate]),model.field(z)[:,coordinate],atol=2e-12,rtol=0)
    model.export(tmp_path/'export.json')
    restored=LocalMemoryROM.from_export(tmp_path/'export.json')
    assert torch.equal(model.field(z),restored.field(z))
    x=model.interface_to_state(z).requires_grad_(True)
    H,e=model.energy_and_gradient(x)
    assert torch.allclose(torch.autograd.grad(H.sum(),x)[0],e,atol=1e-12,rtol=0)
    # A real independent MechanicalROM embedding check, not comparison to the wrapper's own core.
    model.template_mask.zero_()
    z[:,30:]=0
    raw=z[:,:30].numpy().copy(); raw[:,20:]+=base.pm_star
    np.testing.assert_allclose(model.field(z)[:,:30].detach().numpy(),base.field(raw),atol=2e-12,rtol=0)
    model.template_mask.requires_grad_(True)
    with torch.no_grad(): model.template_mask[5]=1
    # R_pi,q must have a one-sided physical gradient at an exactly zero mask.
    loss=model.field(torch.randn(3,model.dim,dtype=torch.float64)*.1).square().sum()
    grad=torch.autograd.grad(loss,model.template_mask)[0]
    assert torch.isfinite(grad[6]) and grad[6].abs()>0
