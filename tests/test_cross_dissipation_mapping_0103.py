import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))

from local_memory_rom_0098 import LocalMemoryROM
from shared_closure_rom_0101 import F23, S23, load_model, make_pair, mapped_coefficients, save_model
from structured_rom import StructuredROM


def _source():
    eye = torch.eye(10, dtype=torch.float64)
    D = torch.zeros((26, 10), dtype=torch.float64)
    D[:9] = eye[:9] - eye[1:]
    D[9] = torch.tensor([1., 0, 0, 0, 0, 0, 0, 0, 0, -1.])
    D[10:] = D[:16].clone()
    base = StructuredROM(
        mass=torch.linspace(1., 2., 10), D=D, coupling=torch.linspace(.4, 1.2, 26),
        delta_star=torch.linspace(-.1, .1, 10), damping=torch.linspace(.05, .14, 10),
        gain=torch.linspace(.8, 1.2, 10), jpi_p=torch.full((10,), .2),
        rpp=torch.full((10,), .15), n_q=0, storage=(),
    ).double()
    model = LocalMemoryROM(base, torch.linspace(.7, 1.4, 20),
                           torch.linspace(.8, 1.2, 10), "L10-S", seed=103)
    with torch.no_grad():
        model.template_mask.copy_(torch.tensor([1., 1., 1., 1., 1., 1., 0., 1.]))
        for i, value in enumerate((.12, .08, .05, .04, .03)):
            model._template_scalars[i].copy_(torch.tensor(value))
        model._template_scalars[5].copy_(torch.tensor(-1.5))
        model._template_scalars[6].copy_(torch.tensor(-1.1))
        model._q_quadratic_raw.copy_(torch.tensor(1.1))
        model.port_raw.copy_(torch.linspace(.01, .06, 6))
    return model


def _as_s23(source, active):
    model = S23.from_config(source.config())
    model.load_state_dict(source.state_dict(), strict=True)
    with torch.no_grad():
        model.template_mask.copy_(source.template_mask)
        model.template_mask[6] = float(active)
    model._template_scalars[6].requires_grad_(active)
    return model


def test_closed_group_preserves_0101_coefficients_and_field_exactly():
    source = _source()
    s_model, f_model = make_pair(source)
    c = source.template_coefficients
    u, v, r = c[3], c[4], c[5]
    a, b, cc = c[:3]
    q_quad = source.q_quadratic
    port = source.port_raw
    expected = torch.stack((
        b, cc, a, u * a + port[1], u * b, u * cc, u * q_quad + port[2],
        a, v * a, v * b, v * cc, v * q_quad,
        u, v, v * a, r * a, r * b, r * cc, r * q_quad,
        port[0], port[3], port[4], port[5],
    ))
    assert torch.equal(mapped_coefficients(s_model), expected)
    z = torch.randn(4, 40, dtype=torch.float64) * .04
    assert torch.allclose(s_model.field(z), f_model.field(z), atol=2e-11, rtol=0)


def test_active_cross_dissipation_matches_field_jacobian_psd_export_and_gradients(tmp_path):
    source = _source()
    s_model = _as_s23(source, active=True)
    f_model = F23(s_model)
    rho = s_model.template_coefficients[6]
    assert rho > 0
    assert torch.allclose(
        f_model.structured._pos("damping"),
        s_model.structured._pos("damping") + rho / 2,
        atol=2e-15, rtol=0,
    )

    z = torch.randn(3, 40, dtype=torch.float64) * .035
    assert torch.allclose(s_model.field(z), f_model.field(z), atol=3e-11, rtol=0)
    point = z[0].detach().requires_grad_(True)
    js = torch.autograd.functional.jacobian(s_model.field, point)
    jf = torch.autograd.functional.jacobian(f_model.field, point)
    assert torch.allclose(js, jf, atol=5e-10, rtol=0)

    _, R = s_model.matrices()
    assert torch.linalg.eigvalsh(R).min() >= -1e-12
    mapped_loss = mapped_coefficients(s_model).square().sum()
    grad = torch.autograd.grad(mapped_loss, s_model._template_scalars[6])[0]
    assert torch.isfinite(grad) and grad.abs() > 0

    s_path = tmp_path / "s-active.json"
    s_payload = s_model.export(s_path)
    assert torch.allclose(torch.tensor(s_payload["rhs_coefficients"], dtype=torch.float64),
                          mapped_coefficients(s_model).detach(), atol=0, rtol=0)
    restored_s = load_model(s_path)
    assert torch.allclose(restored_s.field(z), s_model.field(z), atol=0, rtol=0)
    f_path = tmp_path / "f-active.pt"
    save_model(f_model, f_path)
    restored_f = load_model(f_path)
    assert torch.allclose(restored_f.field(z), f_model.field(z), atol=0, rtol=0)
    assert torch.allclose(restored_s.field(z), restored_f.field(z), atol=3e-11, rtol=0)

