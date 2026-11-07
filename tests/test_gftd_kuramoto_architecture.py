"""Data-free 5N equation and interface qualification, without latent fitting."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments"), str(ROOT / "tests")]
from kiki.models.base import RunContext
from kiki.models.gftd_kuramoto import GFTDKuramotoModel, SO2InvariantHistoryEncoder
from kiki.physics.kuramoto_functional import StructuralGates
from kiki.physics.kuramoto_benchmarks import simulate_partially_observed_kuramoto_benchmark
from r_memory_model_20261025 import RMemoryROM
from test_r_memory_model_20261025 import source

torch.set_num_threads(1)


def parent_fixture(chord=False):
    seed = RMemoryROM(source(), mode="C", seed=1107)
    cfg = seed.config()
    edges = [(i, i+1) for i in range(9)] + [(0, 9)]
    if chord:
        edges.append((0, 5))
    incidence = np.zeros((len(edges), 10))
    for row, (i, j) in enumerate(edges):
        incidence[row, i], incidence[row, j] = 1., -1.
    cfg.update(mass=np.linspace(1., 2., 10).tolist(), D=incidence.tolist(),
               coupling=np.linspace(.3, .9, len(edges)).tolist(),
               base_coupling=np.linspace(.3, .9, len(edges)).tolist())
    for key in ("backbone_raw", "neighbor_A", "active_edge"):
        cfg.pop(key, None)
    parent = RMemoryROM(cfg, mode="C", seed=1107)
    with torch.no_grad():
        parent.kappa_raw.fill_(np.log(.7 / float(parent.kappa_reference)))
        parent.lambda_raw.fill_(np.log(.3))
        parent.local_head.weight.copy_(torch.linspace(-.3, .4, 8).reshape(1, 8))
    return parent


def legal_history(n=10):
    t = np.linspace(-.1, 0., 11)
    return dict(theta_history=np.outer(t*t, np.linspace(.1, .2, n)),
                omega_history=np.outer(2*t, np.linspace(.1, .2, n)),
                history_times=t, p0=np.linspace(-.05, .07, n))


def rk4(function, x, dt):
    a = function(x)
    b = function(x + dt*a/2)
    c = function(x + dt*b/2)
    d = function(x + dt*c)
    return x + dt*(a+2*b+2*c+d)/6


@pytest.mark.parametrize("chord", [False, True])
def test_parent_rhs_energy_initialization_and_multistep(chord):
    parent = parent_fixture(chord)
    model = GFTDKuramotoModel.from_parent(parent)
    rng = np.random.default_rng(1107)
    states = np.concatenate((rng.uniform(-.2, .2, (8, 10)), rng.uniform(-.1, .1, (8, 30))), axis=1)
    def map_state(state):
        d, w, p, r = np.split(state, 4)
        return model.functional.pack_state(d+model.delta_star, w, p, np.zeros(10), r)
    def observed(state):
        theta, w, p, a, r = model.functional.split_state(state)
        np.testing.assert_array_equal(a, np.zeros(10))
        return np.concatenate((theta-model.delta_star, w, p, r))
    for state in states:
        tensor = torch.tensor(state)
        mapped = map_state(state)
        gt, gw, gp, ga, gr = model.functional.split_state(model.functional.rhs(mapped))
        np.testing.assert_allclose(np.concatenate((gt,gw,gp,gr)), parent.field(tensor).detach().numpy(),
                                   atol=1e-12, rtol=1e-11)
        assert np.array_equal(ga, np.zeros(10))
        np.testing.assert_allclose(model.functional.hamiltonian(mapped),
            parent.energy(parent.interface_to_state(tensor)).detach().numpy(), atol=1e-12, rtol=1e-11)
        for dt in (.01, .005):
            x, y = tensor.clone(), mapped.copy()
            with torch.no_grad():
                for _ in range(100):
                    x = parent.rk4(x, dt)
                    y = rk4(model.functional.rhs, y, dt)
                    np.testing.assert_allclose(observed(y), x.numpy(), atol=1e-10, rtol=1e-9)
    for history in (legal_history(), dict(theta_history=np.zeros((11,10)),
            omega_history=np.zeros((11,10)), history_times=np.linspace(-.1,0,11), p0=np.zeros(10))):
        physical = np.zeros((11,30))
        physical[:,:10] = history["theta_history"] - model.delta_star
        physical[:,10:20] = history["omega_history"]
        physical[-1,20:] = history["p0"]
        expected = parent.encode(physical).detach().numpy()
        actual = model.initialize(history)
        np.testing.assert_allclose(actual[40:], expected, atol=1e-12, rtol=1e-11)
        np.testing.assert_array_equal(actual[20:30], history["p0"])
        physical[:-1,20:] = 1000.
        np.testing.assert_array_equal(expected, parent.encode(physical).detach().numpy())
        original = np.concatenate((physical[-1,:20], history["p0"], expected))
        with torch.no_grad():
            for _ in range(100):
                original = parent.rk4(torch.tensor(original), .01).numpy()
        predicted = model.rollout_state(history, np.arange(101)*.01)[-1]
        np.testing.assert_allclose(observed(predicted), original, atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize("modulation", [0., .5])
def test_independent_autograd_and_finite_difference_power(modulation):
    model = GFTDKuramotoModel.from_parent(parent_fixture())
    f = model.functional
    f.gates = StructuralGates(f.gates.s_edge, f.gates.s_edge*modulation, np.ones(10))
    f.gamma_amp[:] = .5
    f.tau_amp[:] = 5.
    rng = np.random.default_rng(1107)
    for _ in range(8):
        state = rng.uniform(-.1, .1, 50)
        state[:10] += model.delta_star
        x = torch.tensor(state, requires_grad=True)
        theta, w, p, a, r = x.split(10)
        B = torch.tensor(f.incidence)
        eta = torch.tensor(f.eta_edge)
        phase = B.T @ theta
        h = B.abs().T @ (torch.tensor(f.gamma_amp) * a)
        modulation_kernel = (1-torch.tensor(f.gates.s_mod)) + torch.tensor(f.gates.s_mod)*h.exp()
        U = (torch.tensor(f.gates.s_edge*f.k_edge)*modulation_kernel *
             (eta.cos()-phase.cos()-eta.sin()*(phase-eta))).sum()
        H = U + .5*((torch.tensor(f.mass)*w*w).sum() + (torch.tensor(f.storage_p)*p*p).sum()
                    +(torch.tensor(f.lambda_amp)*a*a).sum()+(torch.tensor(f.lambda_mem)*r*r).sum())
        gradient = torch.autograd.grad(H, x)[0].detach().numpy()
        np.testing.assert_allclose(f.pack_state(*f.gradients(state)), gradient, atol=1e-11, rtol=1e-10)
        fd = np.empty(50)
        for i in range(50):
            shift = np.eye(50)[i]*1e-5
            fd[i] = (f.hamiltonian(state+shift)-f.hamiltonian(state-shift))/2e-5
        np.testing.assert_allclose(gradient, fd, atol=1e-7, rtol=1e-6)
        audit = f.power_balance_audit(state)
        np.testing.assert_allclose(gradient @ f.rhs(state), audit["expected_dH_dt"], atol=1e-11, rtol=1e-10)
        assert audit["strictly_dissipative"]


def test_old_clipped_counterexample_now_has_actual_gradient():
    model = GFTDKuramotoModel(n_nodes=2, mass=np.array([1.,2.]))
    f = model.functional
    f.gates = StructuralGates.overcomplete(1,2)
    f.gamma_amp[:] = 1.
    f.eta_edge[:] = .4
    state = f.pack_state([2*np.pi+.4,0], [0,0], [0,0], [21,21], [0,0])
    direction = f.rhs(state)
    direction /= np.linalg.norm(direction)
    fd = (f.hamiltonian(state+1e-4*direction)-f.hamiltonian(state-1e-4*direction))/2e-4
    analytic = f.pack_state(*f.gradients(state)) @ direction
    assert fd < 0
    np.testing.assert_allclose(fd, analytic, rtol=1e-7)
    assert f.power_balance_audit(state)["strictly_dissipative"]


def test_so2_and_diagonal_gauge_equivalence():
    model = GFTDKuramotoModel(n_nodes=3)
    f = model.functional
    f.lambda_amp[:] = [2.,3.,4.]
    f.lambda_mem[:] = [.7,1.,2.]
    state = np.linspace(-.2,.3,15)
    shifted = state.copy()
    shifted[:3] += 3.7
    np.testing.assert_allclose(f.rhs(state),f.rhs(shifted),atol=1e-14)
    canonical, transformed = f.canonical_gauge_transform(state)
    np.testing.assert_allclose(f.hamiltonian(state),canonical.hamiltonian(transformed),atol=1e-14)
    expected = f.rhs(state)
    expected[9:12] *= np.sqrt(f.lambda_amp)
    expected[12:] *= np.sqrt(f.lambda_mem)
    np.testing.assert_allclose(expected,canonical.rhs(transformed),atol=1e-14)
    encoder = SO2InvariantHistoryEncoder(3, np.ones(3))
    history = legal_history(3)
    a = encoder.encode(history["theta_history"],history["omega_history"],history["p0"])
    b = encoder.encode(history["theta_history"]+3.7,history["omega_history"],history["p0"])
    np.testing.assert_allclose(a,b,atol=1e-14)


@pytest.mark.parametrize("bad", ["missing_p", "short", "nan", "times", "future"])
def test_initial_input_rejection(bad):
    model = GFTDKuramotoModel(n_nodes=3)
    history = legal_history(3)
    if bad == "missing_p":
        history.pop("p0")
    elif bad == "short":
        history["theta_history"] = history["theta_history"][:1]
    elif bad == "nan":
        history["p0"][0] = np.nan
    elif bad == "times":
        history["history_times"] *= .5
    else:
        history["future_truth"] = np.ones((100,3))
    with pytest.raises(ValueError):
        model.initialize(history)


@pytest.mark.parametrize("times", [[], [1.], [0.,-.01], [0.,.02], [0.,np.nan], [[0.]]])
def test_future_grid_rejection(times):
    with pytest.raises(ValueError):
        GFTDKuramotoModel(n_nodes=3).rollout_state(legal_history(3),times)


def test_roundtrip_all_rhs_parameters_and_parent_initializer(tmp_path):
    model = GFTDKuramotoModel.from_parent(parent_fixture(True))
    f = model.functional
    f.lambda_amp[:] = np.linspace(.4, 1.2, 10)
    f.tau_amp[:] = np.linspace(.3,.9,10)
    f.gate_weight_amp = np.full((10,3),.23)
    f.gate_bias_amp = np.linspace(-.2,.2,10)
    f.gate_weight_mem = np.full((10,3),-.13)
    f.gate_bias_mem = np.linspace(.1,.3,10)
    model.encoder.w_cand_mem *= 2.
    checkpoint = tmp_path/"model.json"
    model.save(checkpoint)
    restored = GFTDKuramotoModel.load(checkpoint)
    state = np.linspace(-.1,.2,50)
    np.testing.assert_array_equal(f.rhs(state),restored.functional.rhs(state))
    np.testing.assert_array_equal(model.initialize(legal_history()),restored.initialize(legal_history()))
    np.testing.assert_array_equal(model.rollout_state(legal_history(),np.arange(11)*.01),
                                  restored.rollout_state(legal_history(),np.arange(11)*.01))
    checkpoint.write_text(json.dumps({"n_nodes":10}))
    with pytest.raises(ValueError,match="4N"):
        GFTDKuramotoModel.load(checkpoint)


def test_empty_fit_and_legacy_modulated_initializer_fail_explicitly():
    model = GFTDKuramotoModel(n_nodes=3)
    with pytest.raises(ValueError):
        model.fit([], [], RunContext())
    model.gates = StructuralGates.overcomplete(3,3)
    model.functional.gates = model.gates
    with pytest.raises(NotImplementedError):
        model.initialize(legal_history(3))


def test_artificial_benchmark_has_independent_mechanical_power():
    dataset = simulate_partially_observed_kuramoto_benchmark(n_nodes=3,n_cases=1,future_steps=1)
    case = dataset.cases[0]
    assert case["ground_truth_latent"]["p_full"].shape == (12,3)
    np.testing.assert_array_equal(case["observed"]["p0"],case["ground_truth_latent"]["p_full"][10])
