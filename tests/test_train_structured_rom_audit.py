"""Synthetic-only audit contracts for Loop0078 pruning/complexity."""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from test_train_structured_rom import source_fixture, screen


def _windows(source, n=4):
    return torch.as_tensor(np.random.default_rng(88).normal(scale=.02, size=(n, 11, 30)), dtype=torch.float64)


def test_c_prune_threshold_is_full_normalized_33_state_field():
    _, source = source_fixture(); model = screen.CFreeROM(source); windows = _windows(source)
    opt = screen.optimizer_for(model); state = torch.cat((windows[:, -1], model.encode(windows)), -1)
    expected = 1e-3 * screen._rms(screen._internal_normalized(model.field(state), model))
    report = model.prune(windows, opt)
    assert report['threshold'] == pytest_approx(float(expected))


def test_g_prune_uses_actual_signed_gamma_and_common_r_direction():
    base, source = source_fixture(); model = screen.GROM(base, source, 'G-auto'); windows = _windows(source)
    model.add_storage(screen.PrimitiveSpec('product', (0, 10), 1e-3, (1., 1.)), 1.)
    model.add_dissipation_vector(np.r_[base.mass, np.zeros(13)], 1e-3)
    with torch.no_grad(): model.structured.storage_gamma[-1].fill_(.23)
    x = model.structured.interface_to_state(torch.cat((windows[:, -1], model.encode(windows)), -1)); J, R = model.structured.matrices()
    h = screen._h_field(model, x, model.structured.storage_specs[-1], J, R)
    h_value=screen._rms((h * model.structured.storage_gamma[-1]) / screen._selection_scale(model,x))
    a = model.structured.dissipation_a[-1]
    assert torch.allclose(a[:10], model.mass) and torch.count_nonzero(a[10:]) == 0
    n=model.structured.n; aa=torch.zeros(model.structured.dim,dtype=x.dtype); aa[n:]=a/torch.linalg.vector_norm(a)
    rfield=-(model.structured.energy_and_gradient(x)[1]@aa)[...,None]*aa*torch.nn.functional.softplus(model.structured.dissipation_rho[-1])
    r_value=screen._rms(rfield / screen._selection_scale(model,x))
    sample=({'id':'synthetic','x':np.zeros((32,30))},10); report=screen.prune_g(model,[],np.random.default_rng(5),screen.optimizer_for(model),windows=([sample],windows))
    got={(r['kind'],r['index']):r['normalized_rms'] for r in report['groups_before']}
    assert got[('H',0)] == pytest_approx(float(h_value))
    assert got[('R',0)] == pytest_approx(float(r_value))


def test_port_prune_freezes_zero_coefficients_but_direction_stays_valid(tmp_path):
    base, source = source_fixture(); model = screen.GROM(base, source, 'G-auto'); windows = _windows(source)
    opt = screen.optimizer_for(model)
    with torch.no_grad():
        for p in model.structured.port_coefficient: p.zero_()
    sample = ({'id':'synthetic','x':np.zeros((32,30))}, 10)
    report = screen.prune_g(model, [], np.random.default_rng(3), opt, windows=([sample], windows[:1]))
    assert report['pruned_ports'] == [0, 1]
    for p, d in zip(model.structured.port_coefficient, model.structured.port_direction):
        assert torch.count_nonzero(p) == 0 and not p.requires_grad and not d.requires_grad
        assert torch.linalg.vector_norm(d) > 0
    torch.save(model.state_dict(),tmp_path/'model.pt'); model.structured.export(tmp_path/'structured_numeric_export.json')
    (tmp_path/'checkpoint.json').write_text(__import__('json').dumps({'arm':'G-auto','source':'unused','extra_groups':model.extra_groups,'requires_grad':{n:bool(p.requires_grad) for n,p in model.named_parameters()}}))
    loaded=screen.load_model(tmp_path,base,source); z=torch.as_tensor(np.random.default_rng(5).normal(size=(2,33)),dtype=torch.float64)
    assert torch.allclose(model.field(z),loaded.field(z),atol=1e-12,rtol=0)


def test_g_complexity_is_fullfield_normalized_port_weighted_and_zero_safe():
    base, source = source_fixture(); model = screen.GROM(base, source, 'G-auto'); z = torch.as_tensor(np.random.default_rng(6).normal(scale=.03,size=(3,33)),dtype=torch.float64)
    value = model.complexity(z)
    x = model.structured.interface_to_state(z); pv = model.structured.port_values(x); B = model.structured.port_matrix()
    expected = torch.stack([5 * screen._rms(screen._internal_normalized(model.structured.state_to_interface(pv[:,j,None]*B[:,j]), model)) for j in range(2)]).mean()
    assert torch.isfinite(value) and torch.allclose(value, expected, atol=1e-12, rtol=0)
    value.backward()
    for p in list(model.structured.port_coefficient)+list(model.structured.port_direction): assert p.grad is not None and torch.isfinite(p.grad).all()
    assert torch.isfinite(model.complexity(torch.zeros_like(z)))


def pytest_approx(x):
    import pytest
    return pytest.approx(x, abs=1e-12, rel=0)
