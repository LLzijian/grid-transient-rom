"""Random fixtures only: exact equations, derivatives and runtime lifecycle."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'experiments'), str(Path(__file__).resolve().parents[1]/'src')]
from closure_training_runtime import TrainingClosureROM
from linear_cache_runtime_20261019 import enable_linear_cache
from local_memory_rom_0098 import LocalMemoryROM
from structured_rom import StructuredROM
from symbolic_closure_rom import SymbolicClosureROM, numpy_field, term_key
from quadratic_memory_runtime_20261020 import QuadraticMemoryRuntime, build_variant, TERMS, QUADRATIC_TERM

DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


def source(device='cpu', linear=False):
    torch.manual_seed(9201)
    D = torch.eye(10, dtype=torch.float64)-torch.eye(10, dtype=torch.float64).roll(1, 1)
    base = StructuredROM(mass=torch.linspace(1.,2.,10), D=D,
        coupling=torch.linspace(.4,1.2,10), delta_star=torch.linspace(-.17,.23,10),
        damping=torch.full((10,),.08), gain=torch.linspace(.8,1.2,10),
        jpi_p=torch.full((10,),.2), rpp=torch.full((10,),.15), n_q=0, storage=()).double()
    old = LocalMemoryROM(base, input_scale=torch.linspace(.7,1.4,20), p_scale=torch.linspace(.8,1.2,10), seed=98)
    with torch.no_grad():
        old.template_mask.copy_(torch.tensor([1.,1.,1.,1.,1.,1.,1.,1.]))
    model = TrainingClosureROM.from_source(old).with_terms(TERMS['A']).to(device)
    with torch.no_grad():
        model.coefficients.copy_(torch.linspace(-.12,.16,8,dtype=torch.float64,device=device))
    return enable_linear_cache(model) if linear else model


def generic(model):
    reference = SymbolicClosureROM.from_config(copy.deepcopy(model).cpu()._export_payload()['config'], terms=model.terms)
    reference.load_state_dict(model.state_dict())
    flags = {n:p.requires_grad for n,p in model.named_parameters()}
    for n,p in reference.named_parameters(): p.requires_grad_(flags[n])
    return reference.to(model.mass.device)


def set_b(model, value):
    i = next(i for i,t in enumerate(model.terms) if term_key(t) == term_key(QUADRATIC_TERM))
    with torch.no_grad(): model.coefficients[i].fill_(value)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('linear', [False, True])
def test_variants_preserve_source_and_seven_coefficients(device, linear):
    original = source(device,linear)
    state = {k:v.clone() for k,v in original.state_dict().items()}
    a,b,c = [build_variant(original,variant) for variant in ('A','B','C')]
    z = torch.randn(4,40,dtype=torch.float64,device=device)*.12
    torch.testing.assert_close(a.field(z),original.field(z),rtol=2e-12,atol=2e-12)
    torch.testing.assert_close(b.field(z),c.field(z),rtol=0,atol=0)
    retained = {term_key(t):v for t,v in zip(original.terms,original.coefficients)}
    for model in (a,b,c):
        for term,coefficient in zip(model.terms,model.coefficients):
            key = term_key(term)
            if key in retained: assert torch.equal(coefficient,retained[key])
            else: assert coefficient.item() == 0.
    for k,v in original.state_dict().items(): assert torch.equal(v,state[k])
    assert all(id(p)!=id(q) for p,q in zip(a.parameters(),original.parameters()))


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('b_value', [.27, -.19])
def test_nonzero_c_field_numpy_state_and_parameter_gradients(device,b_value):
    model = build_variant(source(device),'C'); set_b(model,b_value)
    reference = generic(model)
    z = (torch.randn(3,40,dtype=torch.float64,device=device)*.17).requires_grad_()
    zr = z.detach().clone().requires_grad_()
    u = torch.randn(3,10,dtype=torch.float64,device=device)*.01
    for external in (None,u):
        torch.testing.assert_close(model.field(z,external),reference.field(zr,external),rtol=2e-11,atol=2e-11)
        np.testing.assert_allclose(model.field(z,external).detach().cpu().numpy(),
            numpy_field(copy.deepcopy(reference).cpu(),z.detach().cpu().numpy(),None if external is None else external.cpu().numpy()),rtol=2e-10,atol=2e-10)
    model.begin_graph_cache()
    try:
        fast = model.field(z)
        fast.square().sum().backward()
    finally: model.end_graph_cache()
    reference.field(zr).square().sum().backward()
    torch.testing.assert_close(z.grad,zr.grad,rtol=2e-9,atol=2e-9)
    for (n,p),(nr,q) in zip(model.named_parameters(),reference.named_parameters()):
        assert n == nr and p.requires_grad == q.requires_grad
        assert (p.grad is None) == (q.grad is None), n
        if p.grad is not None: torch.testing.assert_close(p.grad,q.grad,rtol=2e-9,atol=2e-9,msg=n)


@pytest.mark.parametrize('device', DEVICES)
def test_nonzero_b_zero_origin_gradient_hessian_and_jacobian(device):
    src = source(device)
    b = build_variant(src,'B'); c = build_variant(src,'C'); set_b(c,.41)
    x = torch.zeros(40,dtype=torch.float64,device=device,requires_grad=True)
    def added_energy(state):
        return c.energy_and_gradient(state)[0]-b.energy_and_gradient(state)[0]
    grad = torch.autograd.functional.jacobian(added_energy,x)
    hessian = torch.autograd.functional.hessian(added_energy,x)
    torch.testing.assert_close(grad,torch.zeros_like(grad),rtol=0,atol=2e-12)
    torch.testing.assert_close(hessian,torch.zeros_like(hessian),rtol=0,atol=2e-12)
    torch.testing.assert_close(torch.autograd.functional.jacobian(c.field,x),
                               torch.autograd.functional.jacobian(b.field,x),rtol=0,atol=2e-12)
    z = torch.randn(2,40,dtype=torch.float64,device=device)*.2
    assert torch.max(torch.abs(c.field(z)-b.field(z))) > 1e-8
    internal = c.interface_to_state(z).detach().requires_grad_()
    energy, compiled_gradient = c.energy_and_gradient(internal)
    torch.testing.assert_close(torch.autograd.grad(energy.sum(),internal)[0],compiled_gradient,rtol=2e-11,atol=2e-11)


@pytest.mark.parametrize('device', DEVICES)
def test_export_reload_rollout_single_initialization_and_cache(tmp_path,monkeypatch,device):
    model = build_variant(source(device),'C'); set_b(model,.21)
    reference = generic(model)
    h = torch.randn(2,11,30,dtype=torch.float64,device=device)*.025
    calls = []; original_encode = model.encode
    def encode(history):
        calls.append(tuple(history.shape)); return original_encode(history)
    monkeypatch.setattr(model,'encode',encode)
    model.begin_graph_cache()
    with pytest.raises(RuntimeError): model.begin_graph_cache()
    with pytest.raises(RuntimeError): model.with_terms(TERMS['B'])
    with pytest.raises(RuntimeError): model.export(tmp_path/'bad.json')
    actual = model.rollout(h,3,.01)
    model.end_graph_cache()
    assert calls == [(2,11,30)]
    torch.testing.assert_close(actual,reference.rollout(h,3,.01),rtol=2e-11,atol=2e-11)
    monkeypatch.setattr(model,'encode',original_encode)
    model.export(tmp_path/'model.json')
    loaded = QuadraticMemoryRuntime.from_export(tmp_path/'model.json').to(device)
    z = actual[:,-1].detach()
    torch.testing.assert_close(loaded.field(z),model.field(z),rtol=0,atol=0)
    model.begin_graph_cache(); model.load_state_dict(loaded.state_dict())
    assert model._runtime_cache is None and model._quadratic_cache is None
    model.begin_graph_cache(); model.to(device)
    assert model._runtime_cache is None and model._quadratic_cache is None
    set_b(model,.33)
    torch.testing.assert_close(model.field(z),generic(model).field(z),rtol=2e-11,atol=2e-11)
    assert model.with_terms(TERMS['B'])._quadratic_cache is None


def test_strict_structure_guard_and_source_requirements():
    src = source()
    with pytest.raises(ValueError): build_variant(src,'D')
    with pytest.raises(ValueError): build_variant(src.with_terms(TERMS['B']),'C')
    model = build_variant(src,'C')
    with pytest.raises(ValueError): model.with_terms(TERMS['C']+TERMS['B'][:1])
    with pytest.raises(ValueError): model.field(torch.zeros(39,dtype=torch.float64))
    model._term_kinds[-1] = 'port_w'
    with pytest.raises(ValueError): model.field(torch.zeros(40,dtype=torch.float64))
    src.begin_graph_cache()
    try:
        with pytest.raises(RuntimeError): build_variant(src,'C')
    finally: src.end_graph_cache()
