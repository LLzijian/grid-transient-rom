import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))

from local_memory_rom_0098 import LocalMemoryROM
from structured_rom import StructuredROM
from shared_closure_rom_0101 import (
    CLOSURE_NAMES,
    F23,
    S23,
    load_model,
    make_pair,
    mapped_coefficients,
    save_model,
)


def _base():
    eye = torch.eye(10, dtype=torch.float64)
    D = torch.zeros((26, 10), dtype=torch.float64)
    D[:9] = eye[:9] - eye[1:]
    D[9] = torch.tensor([1., 0, 0, 0, 0, 0, 0, 0, 0, -1.])
    D[10:] = D[:16].clone()
    return StructuredROM(
        mass=torch.linspace(1., 2., 10), D=D, coupling=torch.linspace(.4, 1.2, 26),
        delta_star=torch.linspace(-.1, .1, 10), damping=torch.full((10,), .1),
        gain=torch.linspace(.8, 1.2, 10), jpi_p=torch.full((10,), .2),
        rpp=torch.full((10,), .15), n_q=0, storage=(),
    ).double()


def _source():
    source = LocalMemoryROM(_base(), torch.linspace(.7, 1.4, 20),
                            torch.linspace(.8, 1.2, 10), "L10-S", seed=98)
    with torch.no_grad():
        source.template_mask.copy_(torch.tensor([1., 1., 1., 1., 1., 1., 0., 1.]))
        source._template_scalars[0].copy_(torch.tensor(.12))
        source._template_scalars[1].copy_(torch.tensor(.08))
        source._template_scalars[2].copy_(torch.tensor(.05))
        source._template_scalars[3].copy_(torch.tensor(.04))
        source._template_scalars[4].copy_(torch.tensor(.03))
        source._template_scalars[5].copy_(torch.tensor(-1.5))
        source._q_quadratic_raw.copy_(torch.tensor(1.1))
        source.port_raw.copy_(torch.linspace(.01, .06, 6))
    return source


def test_pair_field_encoder_and_single_state_jacobian_match():
    source = _source()
    s_model, f_model = make_pair(source)
    assert isinstance(s_model, S23) and isinstance(f_model, F23)
    assert s_model.dim == f_model.dim == 40
    assert s_model.rank == f_model.rank == 10
    assert tuple(f_model.closure_names) == CLOSURE_NAMES
    assert torch.allclose(f_model.closure_coefficients, mapped_coefficients(s_model))

    z = torch.randn(3, 40, dtype=torch.float64) * .05
    h = torch.randn(3, 11, 30, dtype=torch.float64) * .05
    assert torch.allclose(s_model.encode(h), f_model.encode(h), atol=1e-13, rtol=0)
    assert torch.allclose(s_model.field(z), f_model.field(z), atol=2e-11, rtol=0)

    z1 = z[0].detach().requires_grad_(True)
    js = torch.autograd.functional.jacobian(s_model.field, z1)
    jf = torch.autograd.functional.jacobian(f_model.field, z1)
    assert js.shape == (40, 40) and torch.isfinite(jf).all()
    assert torch.allclose(js, jf, atol=3e-10, rtol=0)


def test_gauge_kinematic_permutation_and_no_unused_closure_modules():
    source = _source()
    s_model, f_model = make_pair(source)
    z = torch.randn(2, 40, dtype=torch.float64) * .03
    shift = torch.full((2, 40), .17, dtype=torch.float64)
    shifted = z.clone()
    shifted[..., :10] += .17
    assert torch.allclose(f_model.field(shifted)[..., :10], f_model.field(z)[..., :10], atol=1e-10)
    assert torch.allclose(f_model.field(z)[..., :10], z[..., 10:20], atol=1e-10)

    names = [name for name, p in f_model.named_parameters() if p.requires_grad]
    assert any(name == "closure_coefficients" for name in names)
    assert not any("_template_scalars" in name or "port_raw" in name or "q_quadratic" in name
                   for name in names)
    assert not hasattr(f_model, "energy") and not hasattr(f_model, "matrices")
    assert len(f_model.closure_coefficients) == 23

    # Machine permutation is applied to all machine coordinates and graph
    # columns/rows; the direct F representation follows the same equivariance.
    perm = torch.tensor([3, 0, 7, 1, 9, 2, 8, 4, 6, 5])
    other = copy.deepcopy(source)
    other.base_D = source.base_D[:, perm]
    other.mass = source.mass[perm]
    other.p_scale = source.p_scale[perm]
    other.delta_star = source.delta_star[perm]
    other.input_scale_delta = source.input_scale_delta[perm]
    other.input_scale_omega = source.input_scale_omega[perm]
    other.input_scale = torch.cat((other.input_scale_delta, other.input_scale_omega))
    other.rate = source.rate[perm]
    other.gain = source.gain[perm]
    other.neighbor_A = source.neighbor_A[perm][:, perm]
    other.port_s_weight = source.port_s_weight[:, perm]
    other.port_c_weight = source.port_c_weight[:, perm]
    other.structured.mass = source.structured.mass[perm]
    other.structured.delta_star = source.structured.delta_star[perm]
    other.structured.gain = source.structured.gain[perm]
    with torch.no_grad():
        other.structured.jpi_p.copy_(source.structured.jpi_p.detach()[perm])
        other.structured.rpp.copy_(source.structured.rpp.detach()[perm])
        other.structured.storage_base.copy_(torch.cat((source.structured.storage_base.detach()[:10][perm],
                                                        source.structured.storage_base.detach()[10:])))
        other.structured.damping.copy_(source.structured.damping.detach()[perm])
        other.structured.D.copy_(source.structured.D[:, perm])
        other.structured.phi.copy_(other.structured.D @ other.structured.delta_star)
        other.phi.copy_(other.base_D @ other.delta_star)
    ps, pf = make_pair(other)
    zp = torch.cat((z[..., :10][..., perm], z[..., 10:20][..., perm],
                    z[..., 20:30][..., perm], z[..., 30:40][..., perm]), dim=-1)
    expected = f_model.field(z)
    observed = pf.field(zp)
    assert torch.allclose(observed, torch.cat((expected[..., :10][..., perm],
                                               expected[..., 10:20][..., perm],
                                               expected[..., 20:30][..., perm],
                                               expected[..., 30:40][..., perm]), dim=-1),
                          atol=2e-9, rtol=0)


def test_gradients_and_directory_exports_reload(tmp_path):
    s_model, f_model = make_pair(_source())
    h = torch.randn(2, 11, 30, dtype=torch.float64) * .02
    z = torch.randn(2, 40, dtype=torch.float64) * .02
    for model in (s_model, f_model):
        loss = model.field(z).square().mean() + model.encode(h).square().mean() + model.complexity()
        params = [p for p in model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        assert grads and all(g is None or torch.isfinite(g).all() for g in grads)
        assert any(g is not None and torch.any(g.abs() > 0) for g in grads)

    for model, name in ((s_model, "s"), (f_model, "f")):
        path = tmp_path / f"{name}.pt"
        save_model(model, path)
        restored = load_model(path)
        assert restored.model_kind == model.model_kind
        assert torch.allclose(restored.field(z), model.field(z), atol=1e-12, rtol=0)
        numeric = tmp_path / f"{name}_numeric.json"
        model.export(numeric)
        restored_json = load_model(numeric)
        assert torch.allclose(restored_json.encode(h), model.encode(h), atol=1e-12, rtol=0)
