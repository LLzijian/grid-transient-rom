"""Data-free, zero-optimizer checks of force-graph mechanics and initialization."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from force_graph_model_20261028 import ForceGraphROM, MODES, PAIRS, RIDGE

torch.set_num_threads(1)
DTYPE = torch.float64
ATOL, RTOL = 1.e-10, 1.e-8


def fixture(unit=False):
    B = torch.zeros(10, 45, dtype=DTYPE)
    for e, (i, j) in enumerate(PAIRS):
        B[i, e], B[j, e] = 1., -1.
    return {"mass": (torch.ones(10) if unit else torch.linspace(.6, 2.1, 10)).tolist(),
            "D": B[:, :26].T.tolist(), "coupling": torch.linspace(.3, .9, 26).tolist(),
            "delta_star": torch.linspace(-.17, .29, 10).tolist(),
            "damping": torch.linspace(.1, .3, 10).tolist(),
            "jpi_p": torch.linspace(.7, 1.6, 10).tolist(),
            "rpp": torch.linspace(.4, .8, 10).tolist(),
            "storage_base": torch.linspace(.7, 1.3, 10).tolist(),
            "gain": torch.linspace(.8, 1.7, 10).tolist(),
            "rate": torch.linspace(.2, .8, 10).tolist(),
            "input_scale_delta": torch.linspace(.1, .3, 10).tolist(),
            "input_scale_omega": torch.linspace(.2, .5, 10).tolist(),
            "p_scale": torch.linspace(.6, 1.4, 10).tolist()}


def history(batch=2):
    generator = torch.Generator().manual_seed(20261028)
    return .06 * torch.randn(batch, 11, 30, dtype=DTYPE, generator=generator)


def state():
    generator = torch.Generator().manual_seed(28101)
    return .15 * torch.randn(3, 40, dtype=DTYPE, generator=generator)


def spatial(model):
    with torch.no_grad():
        if model.mode == "S":
            model.k_bar.add_(torch.linspace(.01, .03, 45, dtype=DTYPE, device=model.mass.device))
        if model.mode == "M":
            model.a_bar.copy_(torch.linspace(.01, .03, 45, dtype=DTYPE, device=model.mass.device))


def explicit_field(model, z):
    d, w, p, r = z.split(10, dim=-1)
    force = torch.zeros_like(d)
    for e, (i, j) in enumerate(PAIRS):
        k = model.kappa_reference * model.k_bar[e] if model.mode == "S" else model.reference_coupling45[e]
        phi = model.delta_star[i] - model.delta_star[j]
        f = k * (torch.sin(phi + d[..., i] - d[..., j]) - torch.sin(phi))
        force[..., i] += f
        force[..., j] -= f
    qr = model.lambda_ * r
    if model.mode == "M":
        increments = torch.zeros_like(r)
        for e, (i, j) in enumerate(PAIRS):
            f = model.lambda_reference * model.a_bar[e] * (r[..., i] - r[..., j])
            increments[..., i] += f
            increments[..., j] -= f
        qr = qr + increments
    return torch.cat((w, (-force - model.damping * w + model.jpi_p * model.storage_base * p - r) / model.mass,
                      -model.jpi_p * w - model.rpp * model.storage_base * p,
                      model.kappa * w - qr), -1)


@pytest.mark.parametrize("unit", (False, True))
def test_same_initial_field_init_and_parameter_counts(unit):
    models = [ForceGraphROM(fixture(unit), mode) for mode in MODES]
    z, h = state(), history()
    for m in models:
        assert m.dim == m.interface_dim == 40
        assert sum(p.numel() for p in m.parameters()) == (2 if m.mode == "L" else 47)
        assert set(dict(m.named_parameters())) == ({"kappa_raw", "lambda_raw"}
            | ({"k_bar"} if m.mode == "S" else {"a_bar"} if m.mode == "M" else set()))
        assert all(not b.requires_grad for b in m.buffers())
        assert torch.allclose(m.field(z), models[0].field(z), atol=1.e-12, rtol=0)
        assert torch.allclose(m.encode(h), models[0].encode(h), atol=1.e-12, rtol=0)
        assert torch.equal(m.initialize(h)[..., :30], h[..., -1, :])
        assert torch.equal(m.kappa_reference, m.mass.log().mean().exp())
        assert torch.equal(m.rstar, m.kappa_reference * m.input_scale_omega)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("unit", (False, True))
def test_explicit_field_matrix_power_and_units(mode, unit):
    m = ForceGraphROM(fixture(unit), mode)
    spatial(m)
    z = state()
    assert torch.allclose(m.field(z), explicit_field(m, z), atol=ATOL, rtol=RTOL)
    x = m.interface_to_state(z).detach().requires_grad_()
    energy, gradient = m.energy_and_gradient(x)
    automatic = torch.autograd.grad(energy.sum(), x, create_graph=True)[0]
    assert torch.allclose(gradient, automatic, atol=ATOL, rtol=0)
    J, R = m.matrices()
    assert torch.equal(J + J.T, torch.zeros_like(J))
    assert torch.linalg.eigvalsh(R).min() >= -1.e-12
    f = m.internal_field(x)
    assert torch.allclose(f, gradient @ (J - R).T, atol=ATOL, rtol=RTOL)
    w, p, r = z[:, 10:20], z[:, 20:30], z[:, 30:]
    base_power = -(m.damping * w.square()).sum(-1) - (m.rpp * (m.storage_base * p).square()).sum(-1)
    qr = m.lambda_ * r
    if mode == "M":
        qr = qr + ((r @ m.B) * m.a_bar * m.lambda_reference) @ m.B.T
    assert ((gradient * f).sum(-1) - base_power + (r * qr).sum(-1) / m.kappa).abs().max() < ATOL
    zero = torch.zeros(40, dtype=DTYPE)
    zero[20:30] = torch.linspace(.1, .3, 10)
    assert torch.allclose(m.field(zero)[10:20], m.jpi_p * m.storage_base * zero[20:30] / m.mass, atol=ATOL, rtol=RTOL)
    zero.zero_(); zero[10:20] = .2
    assert torch.equal(m.field(zero)[:10], zero[10:20])
    assert torch.allclose(m.field(zero)[30:], m.kappa * zero[10:20], atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("mode", ("S", "M"))
def test_zero_edges_have_feasible_direction_gradients_and_projection(mode):
    m = ForceGraphROM(fixture(), mode)
    p = m.k_bar if mode == "S" else m.a_bar
    zeros = p.detach() == 0
    z = state()[0]
    weight = torch.linspace(-.7, .9, 40, dtype=DTYPE).square()
    loss = (m.field(z) * weight).sum()
    loss.backward()
    assert torch.isfinite(p.grad).all() and (p.grad[zeros].abs() > 1.e-12).all()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in m.parameters())
    baseline = loss.detach()
    gradient = p.grad.detach().clone()
    # Every absent edge has a one-sided feasible derivative, no ReLU locking.
    for index in torch.nonzero(zeros).flatten().tolist():
        with torch.no_grad():
            p[index] += 1.e-4
            direction = ((m.field(z) * weight).sum() - baseline) / 1.e-4
            p[index] -= 1.e-4
        assert torch.allclose(direction, gradient[index], atol=ATOL, rtol=RTOL)
    with torch.no_grad():
        p.add_(-.01 * gradient)
    m.project_nonnegative_()
    assert (p >= 0).all()
    assert torch.equal(p.grad, gradient)  # projection does not rewrite gradients/moments


@pytest.mark.parametrize("mode", MODES)
def test_causality_origin_and_common_angle(mode):
    m = ForceGraphROM(fixture(), mode)
    spatial(m)
    h = history()
    shifted = h.clone(); shifted[..., :10] += .37
    assert torch.allclose(m.encode(h), m.encode(shifted), atol=ATOL, rtol=RTOL)
    changed = h.clone(); changed[..., :-1, 20:30] = float("nan")
    assert torch.equal(m.encode(h), m.encode(changed))
    z = state(); zs = z.clone(); zs[..., :10] += .37
    assert torch.allclose(m.field(z), m.field(zs), atol=ATOL, rtol=RTOL)
    origin = torch.zeros(2, 11, 30, dtype=DTYPE)
    assert torch.equal(m.encode(origin), torch.zeros(2, 10, dtype=DTYPE))
    assert torch.equal(m.field(m.initialize(origin)), torch.zeros(2, 40, dtype=DTYPE))
    changed = h.clone(); changed[..., -1, 20:30] += .05
    assert not torch.allclose(m.encode(h), m.encode(changed), atol=1.e-8, rtol=0)


@pytest.mark.parametrize("mode", MODES)
def test_machine_permutation(mode):
    cfg = fixture()
    m = ForceGraphROM(cfg, mode); spatial(m)
    perm = torch.tensor([7, 0, 9, 4, 2, 6, 1, 8, 3, 5])
    other = copy.deepcopy(cfg)
    for key, value in cfg.items():
        if key == "D":
            other[key] = torch.tensor(value)[:, perm].tolist()
        elif len(value) == 10:
            other[key] = torch.tensor(value, dtype=DTYPE)[perm].tolist()
    p = ForceGraphROM(other, mode)
    if mode != "L":
        original = m.k_bar if mode == "S" else m.a_bar
        target = p.k_bar if mode == "S" else p.a_bar
        with torch.no_grad():
            for edge, (i, j) in enumerate(PAIRS):
                target[edge] = original[PAIRS.index(tuple(sorted((int(perm[i]), int(perm[j])))))]
    index40 = torch.cat(tuple(perm + offset for offset in (0, 10, 20, 30)))
    index30 = index40[:30]
    z, h = state(), history()
    assert torch.allclose(p.field(z[..., index40]), m.field(z)[..., index40], atol=ATOL, rtol=RTOL)
    assert torch.allclose(p.encode(h[..., index30]), m.encode(h)[..., perm], atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("mode", MODES)
def test_independent_history_quadrature_and_conditioning(mode):
    m = ForceGraphROM(fixture(), mode); spatial(m)
    h = history()
    # Globally linear omega allows an independent single exponential at each t.
    omega_start = torch.linspace(-.03, .06, 10, dtype=DTYPE)
    slope = torch.linspace(.02, -.04, 10, dtype=DTYPE)
    t = torch.linspace(0., .1, 11, dtype=DTYPE)
    h = h[:1]
    h[0, :, 10:20] = omega_start + t[:, None] * slope
    pstart = torch.linspace(-.04, .03, 10, dtype=DTYPE)
    P = torch.zeros(30, 30, dtype=DTYPE)
    P[:10, :10] = -torch.diag(m.rpp * m.storage_base)
    P[:10, 10:20] = -torch.diag(m.jpi_p)
    P[10:20, 20:] = torch.eye(10, dtype=DTYPE)
    initial = torch.cat((pstart, omega_start, slope))
    phistory = torch.stack([(torch.matrix_exp(P * tick) @ initial)[:10] for tick in t])
    h[0, -1, 20:30] = phistory[-1]
    Q = m.lambda_ * torch.eye(10, dtype=DTYPE)
    if mode == "M":
        for edge, (i, j) in enumerate(PAIRS):
            b = torch.zeros(10, dtype=DTYPE); b[i], b[j] = 1., -1.
            Q = Q + m.lambda_reference * m.a_bar[edge] * torch.outer(b, b)
    V = torch.zeros(30, 30, dtype=DTYPE)
    V[:10, :10], V[:10, 10:20] = -Q, m.kappa * torch.eye(10, dtype=DTYPE)
    V[10:20, 20:] = torch.eye(10, dtype=DTYPE)
    initialv = torch.cat((torch.zeros(10, dtype=DTYPE), omega_start, slope))
    v = torch.stack([(torch.matrix_exp(V * tick) @ initialv)[:10] for tick in t])
    E = torch.stack([torch.matrix_exp(-Q * tick) for tick in t])
    system = m.history_system(h)
    assert torch.allclose(system["p_history"][0], phistory, atol=ATOL, rtol=RTOL)
    assert torch.allclose(system["v_history"][0], v, atol=ATOL, rtol=RTOL)
    assert torch.allclose(system["E"], E, atol=ATOL, rtol=RTOL)
    C = torch.stack([(.02 / 6) * (E[i] + 4 * E[i + 1] + E[i + 2]) for i in range(0, 10, 2)])
    phys = torch.cat((h[0, :, :20], phistory, torch.zeros(11, 10, dtype=DTYPE)), -1)
    fbase = explicit_field(m, phys)[:, 10:20] * m.mass
    b = torch.stack([(.02 / 6) * ((fbase - v)[i] + 4 * (fbase - v)[i + 1] + (fbase - v)[i + 2])
                     - m.mass * (h[0, i + 2, 10:20] - h[0, i, 10:20]) for i in range(0, 10, 2)])
    A = (C * m.rstar[None, None, :] / (.02 * m.rstar[None, :, None])).reshape(50, 10)
    scaled = (b / (.02 * m.rstar)).reshape(50)
    s = torch.linalg.solve(A.T @ A + RIDGE * torch.eye(10, dtype=DTYPE), A.T @ scaled)
    expected = E[-1] @ (m.rstar * s) + v[-1]
    assert torch.allclose(system["C"], C, atol=ATOL, rtol=RTOL)
    assert torch.allclose(system["b"][0], b, atol=ATOL, rtol=RTOL)
    assert torch.allclose(system["r0"][0], expected, atol=ATOL, rtol=RTOL)
    diagnostics = m.initialization_diagnostics(h)
    assert len(diagnostics["design_singular_values"]) == 10
    assert all(0 < x < 1 for x in diagnostics["ridge_singular_direction_retention"])
    assert diagnostics["design_condition_number"] >= 1
    assert "not identifiability of 47" in diagnostics["claim_scope"]


def test_noiseless_history_consistency_with_fixed_ridge():
    m = ForceGraphROM(fixture(), "M"); spatial(m)
    z = .1 * state()[0]
    u = z[30:].clone()
    samples = [z]
    with torch.no_grad():
        m.begin_graph_cache()
        for i in range(1000):
            z = m.rk4(z, .0001)
            if (i + 1) % 100 == 0:
                samples.append(z)
        m.end_graph_cache()
    samples = torch.stack(samples)
    system = m.history_system(samples[:, :30])
    quadrature_defect = (system["C"] @ u - system["b"]).abs().max()
    assert quadrature_defect < 1.e-8
    assert (system["r0"] - samples[-1, 30:]).abs().max() < 5.e-6
    # The estimator includes a fixed ridge and sampled-history quadrature; it
    # does not claim an exact reconstruction of the whole observed trajectory.
    residual = system["s"] @ system["A"].T - system["bscaled"]
    stationarity = residual @ system["A"] + RIDGE * system["s"]
    assert stationarity.abs().max() < ATOL


@pytest.mark.parametrize("mode", MODES)
def test_all_gradients_cache_and_inference_to_train(mode):
    m = ForceGraphROM(fixture(), mode); spatial(m)
    h = history()
    with torch.inference_mode():
        m.begin_graph_cache()
        m.fast_step(m.initialize(h), .01)
    # Changing grad mode automatically discards inference tensors even if a
    # caller forgot to end that cache, and train()/load()/to() also invalidate.
    out = m.fast_step(m.initialize(h), .01)
    loss = (out * torch.linspace(.2, 1.1, 40, dtype=DTYPE)).square().sum()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    assert m._cache is None
    m.zero_grad(set_to_none=True)
    m.begin_graph_cache()
    before = m.field(state()).detach()
    with torch.no_grad():
        m.kappa_raw.add_(.1)
    after = m.field(state())
    assert not torch.equal(before, after)
    m.end_graph_cache()
    assert torch.equal(after, m.field(state()))
    m.begin_graph_cache(); m.train(); assert m._cache is None
    m.begin_graph_cache(); m.cpu(); assert m._cache is None
    m.begin_graph_cache(); m.load_state_dict(m.state_dict()); assert m._cache is None
    # A fresh temporary directory avoids reading a pre-existing pytest user's
    # shared directory, which can be outside this task's filesystem permission.
    with tempfile.TemporaryDirectory(prefix="force-graph-export-") as folder:
        path = Path(folder) / f"{mode}.json"
        m.export(path); loaded = ForceGraphROM.from_export(path)
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in m.state_dict().items())
    assert torch.allclose(m.field(state()), loaded.field(state()), atol=ATOL, rtol=RTOL)
    assert torch.allclose(m.encode(h), loaded.encode(h), atol=ATOL, rtol=RTOL)
    loaded.rollout(h, 2, .01, method="fast").square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in loaded.parameters())


def test_true_rk4_and_split_linear_flow_and_order():
    cfg = fixture(); cfg["coupling"] = [0.] * 26
    m = ForceGraphROM(cfg, "M"); spatial(m)
    z = state()
    # With zero potential the Strang step is the exact entire 40D linear flow.
    context = m._context(); A = m._linear_generator(context)
    expected = m.state_to_interface(m.interface_to_state(z) @ torch.matrix_exp(A * .01).T)
    assert torch.allclose(m.fast_step(z, .01), expected, atol=ATOL, rtol=RTOL)
    m = ForceGraphROM(fixture(), "M"); spatial(m)
    a = explicit_field(m, z); b = explicit_field(m, z + .005 * a)
    c = explicit_field(m, z + .005 * b); d = explicit_field(m, z + .01 * c)
    assert torch.allclose(m.rk4(z, .01), z + (.01 / 6) * (a + 2 * b + 2 * c + d), atol=ATOL, rtol=RTOL)
    with torch.no_grad():
        m.begin_graph_cache()
        reference = z
        for _ in range(160):
            reference = m.rk4(reference, .000625)
        coarse, fine = z, z
        for _ in range(10):
            coarse = m.fast_step(coarse, .01)
        for _ in range(20):
            fine = m.fast_step(fine, .005)
        m.end_graph_cache()
    error_coarse = (coarse - reference).norm()
    error_fine = (fine - reference).norm()
    assert error_coarse > 0 and 3.8 < error_coarse / error_fine < 4.2
    assert not torch.equal(m.fast_step(z, .01), m.rk4(z, .01))


@pytest.mark.parametrize("source,expected_sha", (
    ("private_data/r_memory_20261025/run01_repair01/models/A_25101/checkpoints/step000512/model.json",
     "6710d648b2edc8e4c579480ba76e3282ed03b9647b507f4ef51b21fca0e65636"),
    ("private_data/closure_mechanism_20261019/P2/quartic_memory01/models/6101_N/checkpoints/step001024/model.json",
     "c926caa685f2105d908c5d642f5944cc78f467744b3f9bd24831085eeca700a5")))
def test_real_source_metadata_strict_extraction(source, expected_sha):
    path = ROOT / source
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha
    raw = json.loads(path.read_text())
    m = ForceGraphROM.from_source_export(path, "M")
    assert m.source_identity["source_sha256"] == expected_sha
    for name in ("damping", "jpi_p", "rpp", "storage_base"):
        expected = torch.nn.functional.softplus(torch.tensor(raw["state_dict"]["structured." + name], dtype=DTYPE))
        expected *= torch.tensor(raw["state_dict"]["structured." + name + "_mask"], dtype=DTYPE)
        assert torch.equal(getattr(m, name), expected[:10])
    assert torch.equal(m.mass, torch.tensor(raw["state_dict"]["mass"], dtype=DTYPE))
    assert torch.equal(m.base_D, torch.tensor(raw["state_dict"]["base_D"], dtype=DTYPE))
    z = state()
    sd = raw["state_dict"]
    coupling = torch.nn.functional.softplus(torch.tensor(sd["structured.coupling"], dtype=DTYPE))
    coupling *= torch.tensor(sd["structured.coupling_mask"], dtype=DTYPE)
    phase = m.base_D @ m.delta_star
    grad = ((torch.sin(z[..., :10] @ m.base_D.T + phase) - torch.sin(phase)) * coupling) @ m.base_D
    base = -grad - m.damping * z[..., 10:20] + m.jpi_p * m.storage_base * z[..., 20:30]
    assert torch.allclose(m.base_momentum_field(z[..., :30]), base, atol=ATOL, rtol=RTOL)
    models = [ForceGraphROM.from_source_export(path, mode) for mode in MODES]
    for candidate in models:
        assert torch.allclose(candidate.field(z), m.field(z), atol=1.e-12, rtol=0)
        assert torch.allclose(candidate.encode(history()), m.encode(history()), atol=1.e-12, rtol=0)
    assert not any(any(t in key for t in ("gru", "head", "template", "port", "q_quadratic")) for key, _ in m.named_parameters())
    assert m._source_config["deleted_source_components"]["state_keys"]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable; not replaced by a CPU claim")
def test_cuda_double_field_initialization_step_and_raw_gradients(mode):
    cpu = ForceGraphROM(fixture(), mode); spatial(cpu)
    gpu = ForceGraphROM(fixture(), mode).cuda(); gpu.load_state_dict(cpu.state_dict())
    h, z = history(), state()
    assert torch.allclose(cpu.field(z), gpu.field(z.cuda()).cpu(), atol=ATOL, rtol=RTOL)
    assert torch.allclose(cpu.encode(h), gpu.encode(h.cuda()).cpu(), atol=ATOL, rtol=RTOL)
    assert torch.allclose(cpu.fast_step(z, .01), gpu.fast_step(z.cuda(), .01).cpu(), atol=ATOL, rtol=RTOL)
    for m, data in ((cpu, h), (gpu, h.cuda())):
        m.begin_graph_cache()
        m.fast_step(m.initialize(data), .01).square().mean().backward()
        m.end_graph_cache()
    for (name, p), (gn, g) in zip(cpu.named_parameters(), gpu.named_parameters()):
        assert name == gn and p.grad is not None and g.grad is not None
        assert torch.isfinite(p.grad).all() and torch.isfinite(g.grad).all()
        assert torch.allclose(p.grad, g.grad.cpu(), atol=ATOL, rtol=RTOL)
