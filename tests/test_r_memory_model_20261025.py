"""Data-free checks of the fixed-action memory equations and causal encoder."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from local_memory_rom_0098 import LocalMemoryROM
from r_memory_model_20261025 import FORMAT, RMemoryROM
from structured_rom import StructuredROM

torch.set_num_threads(1)


def source():
    edges = [(i, j) for i in range(10) for j in range(i + 1, 10)][:26]
    D = torch.zeros((26, 10), dtype=torch.float64)
    for k, (i, j) in enumerate(edges):
        D[k, i], D[k, j] = 1., -1.
    base = StructuredROM(
        mass=torch.linspace(.6, 2.1, 10), D=D,
        coupling=torch.linspace(.3, .9, 26),
        delta_star=torch.linspace(-.17, .29, 10),
        damping=torch.linspace(.1, .3, 10), gain=torch.linspace(.8, 1.7, 10),
        jpi_p=torch.linspace(.7, 1.6, 10), rpp=torch.linspace(.4, .8, 10),
        storage_base=torch.linspace(.7, 1.3, 10), n_q=0)
    base.rate = torch.linspace(.2, .8, 10, dtype=torch.float64)
    src = LocalMemoryROM(base, torch.linspace(.1, .8, 20), torch.linspace(.6, 1.4, 10), "L10-S", seed=55)
    with torch.no_grad():
        for i in range(8):
            src.set_active(i, True)
        src.port_raw.copy_(torch.tensor([.12, -.04, .21, .02, -.03, .05]))
        src._template_scalars[0].fill_(.7)
        src._template_scalars[1].fill_(.3)
    return src


def model(mode="B", seed=25101):
    return RMemoryROM(source(), mode=mode, seed=seed, source_identity={"fixture": "data-free"})


def history(batch=2):
    generator = torch.Generator().manual_seed(19)
    return .1 * torch.randn(batch, 11, 30, dtype=torch.float64, generator=generator)


def activate_head(m):
    with torch.no_grad():
        m.local_head.weight.copy_(torch.linspace(-.3, .4, 8, dtype=torch.float64).reshape(1, 8))


def test_strict_m3_extraction_and_no_old_memory_parameters():
    src = source()
    m = RMemoryROM.from_source(src)
    for name in ("coupling", "damping", "jpi_p", "rpp", "storage_base"):
        expected = src.structured._pos(name)
        if name == "storage_base":
            expected = expected[:10]
        assert torch.equal(m.structured._pos(name), expected)
    assert m.structured.n_q == 0
    assert not m.structured.storage_specs and not m.structured.port_specs
    assert not m.structured.exchange_specs and not m.structured.dissipation_specs
    assert not any("template" in n or "port_raw" in n or "q_quadratic" in n for n, _ in m.named_parameters())
    z = torch.linspace(-.2, .3, 80, dtype=torch.float64).reshape(2, 40)
    r = z[:, 30:]
    xb = torch.cat((z[:, :10], z[:, 10:20] * m.mass, z[:, 20:30]), -1)
    fb = m.structured.field(xb)
    actual = m.internal_field(m.interface_to_state(z))
    assert torch.allclose(actual[:, :10], fb[:, :10], atol=1e-14, rtol=0)
    assert torch.allclose(actual[:, 10:20], fb[:, 10:20] - r, atol=1e-14, rtol=0)
    assert torch.allclose(actual[:, 20:30], fb[:, 20:30], atol=1e-14, rtol=0)
    # The donor's q-dependent mechanical/edge/port terms are not retained.
    assert not torch.allclose(src.field(z)[:, :30], m.field(z)[:, :30])


def test_empirical_mechanical_coefficients_and_complete_frequency():
    m = model()
    z = torch.zeros(40, dtype=torch.float64)
    z[20:30] = torch.linspace(.1, .3, 10)
    G = m.structured._pos("jpi_p")
    Sp = m.structured._pos("storage_base")
    assert torch.allclose(m.field(z)[10:20], G * Sp * z[20:30] / m.mass)
    assert not torch.allclose(G * Sp, torch.ones(10, dtype=torch.float64))
    z.zero_()
    z[10:20] = .2
    f = m.field(z)
    assert torch.equal(f[:10], z[10:20])
    assert torch.allclose(f[20:30], -G * z[10:20])
    assert torch.allclose(f[30:], m.kappa * z[10:20])
    assert torch.count_nonzero(f[30:]) == 10  # common frequency is retained


def test_analytic_energy_gradient_and_power_identity():
    m = model("C")
    x = torch.linspace(-.4, .6, 160, dtype=torch.float64).reshape(4, 40).requires_grad_()
    H, e = m.energy_and_gradient(x)
    auto = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
    assert torch.allclose(e, auto, atol=2e-14, rtol=0)
    J, R = m.matrices()
    assert torch.equal(J + J.T, torch.zeros_like(J))
    assert torch.linalg.eigvalsh(R).min() >= 0
    actual = m.internal_field(x)
    assert torch.allclose(actual, e @ (J - R).T, atol=3e-14, rtol=0)
    hb, eb = m.structured.energy_and_gradient(x[:, :30])
    base_power = (eb * m.structured.field(x[:, :30])).sum(-1)
    defect = (e * actual).sum(-1) - base_power + (m.lambda_ / m.kappa) * x[:, 30:].square().sum(-1)
    assert defect.abs().max() < 3e-13
    balance = (e * actual).sum(-1) + (e @ R * e).sum(-1)
    assert balance.abs().max() < 3e-13
    assert m.port_matrix().shape == (40, 0) and m.port_values(x).shape == (4, 0)
    with pytest.raises(ValueError, match="no external ports"):
        m.field(m.state_to_interface(x), u=torch.zeros(4, 10))


def test_pair_has_identical_field_initial_weights_and_parameter_budget():
    src = source()
    b, c = (RMemoryROM(src, mode=mode, seed=17) for mode in ("B", "C"))
    assert b.dim == b.interface_dim == 40
    assert b.complexity_counts()["raw_trainable_scalar_count"] == c.complexity_counts()["raw_trainable_scalar_count"]
    for name, value in b.state_dict().items():
        assert torch.equal(value, c.state_dict()[name]), name
    z = torch.linspace(-.3, .4, 120, dtype=torch.float64).reshape(3, 40)
    assert torch.equal(b.field(z), c.field(z))
    assert torch.equal(b.encode(history()), torch.zeros(2, 10, dtype=torch.float64))
    assert torch.count_nonzero(c.encode(history())) > 0
    expected = b.mass.log().mean().exp()
    assert torch.equal(b.kappa, expected)
    assert torch.equal(b.lambda_, torch.tensor(1., dtype=torch.float64))
    assert torch.equal(b.rstar, expected * b.input_scale_omega)


@pytest.mark.parametrize("mode", ("B", "C"))
def test_causal_features_p0_and_origin_anchor(mode):
    m = model(mode)
    activate_head(m)
    h = history()
    before = m.encode(h)
    changed = h.clone()
    changed[:, :10, 20:30] += 1000.
    assert torch.equal(before, m.encode(changed))
    changed[:, -1, 20:30] += 1.
    if mode == "B":
        assert torch.equal(before, m.encode(changed))
    else:
        expected = m.structured._pos("jpi_p") * m.structured._pos("storage_base")
        assert torch.allclose(m.encode(changed) - before, expected.expand_as(before), atol=2e-13, rtol=0)
    assert torch.equal(m.encode(torch.zeros_like(h)), torch.zeros(2, 10, dtype=torch.float64))
    with pytest.raises(ValueError, match="exactly eleven"):
        m.encode(torch.cat((h, history()[:, :1]), 1))


def test_endpoint_quadratic_derivative_and_no_detach():
    m = model("C")
    t = torch.linspace(-.1, 0., 11, dtype=torch.float64)
    a = torch.linspace(-.2, .1, 10, dtype=torch.float64)
    b = torch.linspace(-.3, .4, 10, dtype=torch.float64)
    h = history()
    h[:, :, 10:20] = a + t[:, None] * b + .73 * t[:, None].square()
    assert torch.allclose(m.endpoint_slope(h), b.expand(2, 10), atol=2e-13, rtol=0)
    h.requires_grad_()
    r = m.encode(h)
    gh, gG, gk = torch.autograd.grad(r.square().sum(), (h, m.structured.jpi_p, m.structured.coupling))
    assert torch.isfinite(gh).all() and torch.count_nonzero(gh[:, :10, 10:20]) > 0
    assert torch.count_nonzero(gh[:, :10, 20:30]) == 0
    assert torch.count_nonzero(gh[:, -1, 20:30]) > 0
    assert torch.isfinite(gG).all() and torch.count_nonzero(gG) > 0
    assert torch.isfinite(gk).all() and torch.count_nonzero(gk) > 0


@pytest.mark.parametrize("mode", ("B", "C"))
def test_angle_translation_scalar_batch_and_finite_backward(mode):
    m = model(mode)
    activate_head(m)
    h = history()
    shifted = h.clone()
    shifted[..., :10] += .73
    assert torch.allclose(m.encode(h), m.encode(shifted), atol=2e-13, rtol=0)
    z = torch.cat((h[:, -1], m.encode(h)), -1)
    zs = z.clone()
    zs[:, :10] += .73
    assert torch.allclose(m.field(z), m.field(zs), atol=2e-13, rtol=0)
    assert torch.allclose(m.field(z)[0], m.field(z[0]), atol=2e-14, rtol=0)
    assert torch.allclose(m.encode(h)[0], m.encode(h[0]), atol=2e-14, rtol=0)
    value = m.rk4(z, .01).square().mean()
    grads = torch.autograd.grad(value, tuple(m.parameters()), allow_unused=True)
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_machine_relabeling_with_matching_fixed_features():
    m = model("C")
    activate_head(m)
    cfg = m.config()
    perm = torch.tensor([5, 2, 9, 0, 8, 1, 7, 3, 6, 4])
    for key in ("mass", "delta_star", "damping", "gain", "jpi_p", "rpp", "storage_base", "rate", "p_scale", "input_scale_delta", "input_scale_omega"):
        cfg[key] = torch.as_tensor(cfg[key], dtype=torch.float64)[perm].tolist()
    cfg["D"] = torch.as_tensor(cfg["D"], dtype=torch.float64)[:, perm].tolist()
    cfg["neighbor_A"] = torch.as_tensor(cfg["neighbor_A"], dtype=torch.float64)[perm][:, perm].tolist()
    for key in ("damping", "jpi_p", "rpp", "storage_base"):
        for suffix in ("", "_mask"):
            cfg["backbone_raw"][key + suffix] = torch.as_tensor(cfg["backbone_raw"][key + suffix], dtype=torch.float64)[perm].tolist()
    other = RMemoryROM(cfg, mode="C", seed=m.seed)
    other.local_gru.load_state_dict(m.local_gru.state_dict())
    other.local_head.load_state_dict(m.local_head.state_dict())
    h = history()
    hp = torch.cat([v[:, :, perm] for v in h.split(10, dim=-1)], -1)
    assert torch.allclose(other.encode(hp), m.encode(h)[:, perm], atol=3e-13, rtol=0)
    z = torch.cat((h[:, -1], m.encode(h)), -1)
    zp = torch.cat([v[:, perm] for v in z.split(10, dim=-1)], -1)
    expected = torch.cat([v[:, perm] for v in m.field(z).split(10, dim=-1)], -1)
    assert torch.allclose(other.field(zp), expected, atol=3e-13, rtol=0)


@pytest.mark.parametrize("mode", ("B", "C"))
def test_export_reload_and_inference_to_training_cache(tmp_path, mode):
    m = model(mode)
    activate_head(m)
    path = tmp_path / "model.json"
    payload = m.export(path)
    assert payload["format"] == payload["model_kind"] == FORMAT
    other = RMemoryROM.from_export(path)
    assert other.source_identity == m.source_identity
    for key, value in m.state_dict().items():
        assert torch.equal(value, other.state_dict()[key]), key
    h = history()
    z = torch.cat((h[:, -1], m.encode(h)), -1)
    assert torch.equal(m.field(z), other.field(z))
    assert torch.equal(m.encode(h), other.encode(h))
    with torch.inference_mode():
        other.begin_graph_cache()
        other.field(z.detach())
    # Switching modes invalidates a cache created from inference tensors.
    loss = other.field(z.detach()).square().sum()
    loss.backward()
    assert other._runtime_cache is None
    other.begin_graph_cache()
    assert other._runtime_cache is not None
    with pytest.raises(RuntimeError, match="ending"):
        other.export(path)
    other.load_state_dict(m.state_dict())
    assert other._runtime_cache is None and other._graph_cache is None
    other.begin_graph_cache()
    other.to("cpu")
    assert other._runtime_cache is None


def test_rollout_encodes_once_and_preserves_shape():
    m = model("C")
    h = history()
    calls = []
    original = m.encode
    def counted(value):
        calls.append(1)
        return original(value)
    m.encode = counted
    path = m.rollout(h, 4, .01)
    assert path.shape == (2, 5, 40) and len(calls) == 1
    assert torch.equal(path[:, 0, :30], h[:, -1])
    with pytest.raises(ValueError):
        m.rollout(h, -1, .01)
