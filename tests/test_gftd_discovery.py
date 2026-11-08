"""Analytic and mock qualification; no registered Run2 conditions or seeds."""
import copy

import numpy as np
import pytest

from kiki.models import gftd_discovery as core
from kiki.physics.kuramoto_functional import GeneratingFunctionalKuramoto, StructuralGates


def functional(n=4):
    edges = n*(n-1)//2
    return GeneratingFunctionalKuramoto(
        mass=np.linspace(1., 2., n), damping=np.full(n, .2), k_edge=np.ones(edges),
        eta_edge=np.linspace(-.03, .04, edges), gamma_amp=np.full(n, .5),
        lambda_amp=np.ones(n), lambda_mem=np.full(n, 1/.7), tau_amp=np.full(n, 5.),
        tau_mem_sync=np.full(n, 1/(.7*.3)), tau_mem_coi=1., j_omega_mem=np.full(n, .7),
        gates=StructuralGates.static_baseline_limit(np.ones(edges), n),
        gain_p=np.linspace(.7, 1.1, n), storage_p=np.linspace(.8, 1.2, n), rate_p=np.full(n, .3))


def observation(n=4):
    times = np.linspace(-.1, 0., 11)
    return dict(theta_history=(times+.1)[:, None]*np.arange(1, n+1)[None, :],
                omega_history=np.tile(np.linspace(.02, .08, n), (11, 1)),
                history_times=times, p_end=np.linspace(.01, .02, n), sigma=1e-4)


@pytest.mark.parametrize("key", ["true_a", "true_r", "future", "case", "k", "beta", "source_identity"])
def test_observation_whitelist_rejects_hidden_information(key):
    packet = observation()
    packet[key] = 0.
    with pytest.raises(ValueError, match="only"):
        core.validate_observation(packet, 4)


def test_observation_validates_shapes_times_scale_and_copies():
    packet = observation(10)
    clean = core.validate_observation(packet, 10)
    clean["p_end"][0] = 99.
    assert packet["p_end"][0] != 99.
    for name, value in (("sigma", -.1), ("sigma", np.nan), ("p_end", np.zeros(4)),
                        ("history_times", np.linspace(0., .1, 11))):
        wrong = dict(packet, **{name: value})
        with pytest.raises(ValueError):
            core.validate_observation(wrong, 10)


def test_boole_polynomials_and_constant_speed_weak_derivative():
    for count in (41, 81):
        t = np.linspace(-.1, 0., count)
        weights = core.boole_weights(t)
        tau = (t-t[0])/.1
        for power in range(6):
            np.testing.assert_allclose(weights@tau**power, .1/(power+1), atol=3e-17, rtol=2e-15)
        v, dv = core.weak_test_functions(t)
        np.testing.assert_allclose(dv@weights, np.zeros(4), atol=3e-16, rtol=0)
        np.testing.assert_allclose(v[:, [0, -1]], 0., atol=0., rtol=0)
    with pytest.raises(ValueError):
        core.boole_weights(np.linspace(-.1, 0., 11))


def test_constant_forcing_ports_include_full_gain_storage_rate_and_memory_gate():
    f = functional()
    f.gates = StructuralGates(s_edge=np.ones(6), s_mod=np.zeros(6), s_mem=np.full(4, .6), s_coi=0.)
    t = np.linspace(-.1, 0., 81)
    w = np.array([.03, -.02, .01, -.04])
    p_end = np.linspace(.01, .04, 4)
    p, rf = core.reconstruct_ports(t, np.tile(w, (len(t), 1)), p_end, f)
    alpha = f.rate_p*f.storage_p
    offset = f.gain_p*w/alpha
    expected_p = (p_end+offset)*np.exp(alpha*(t[-1]-t)[:, None])-offset
    decay = f.lambda_mem/f.tau_mem_sync
    expected_rf = f.gates.s_mem*f.j_omega_mem*w/decay*(1-np.exp(-decay*(t-t[0])[:, None]))
    np.testing.assert_allclose(p, expected_p, atol=2e-16, rtol=2e-14)
    np.testing.assert_allclose(rf, expected_rf, atol=2e-17, rtol=2e-13)
    np.testing.assert_array_equal(p[-1], p_end)
    np.testing.assert_array_equal(rf[0], np.zeros(4))


@pytest.mark.parametrize("unsupported", ["gate", "laplacian", "coi"])
def test_reconstruction_and_weak_cache_reject_unsupported_memory(unsupported):
    f = functional()
    if unsupported == "gate":
        f.gate_weight_mem = np.zeros((4, 3))
    elif unsupported == "laplacian":
        f.mem_laplacian_weights = np.zeros(6)
    else:
        object.__setattr__(f.gates, "s_coi", 1.)
    packet = observation()
    with pytest.raises(ValueError, match="constant diagonal"):
        core.reconstruct_ports(packet["history_times"], packet["omega_history"], packet["p_end"], f)
    with pytest.raises(ValueError, match="constant diagonal"):
        core.prepare_weak_history(packet, f)


@pytest.mark.parametrize("n", [2, 4, 10])
def test_exact_conditional_dictionary_and_independent_amplitude_partial(n):
    f, packet = functional(n), observation(n)
    cache = core.prepare_weak_history(packet, f)
    amplitude = np.tile(np.linspace(-.1, .15, n), (len(cache["t"]), 1))
    result = core.conditional_dictionary(cache, f, amplitude, a_times=cache["t"])
    edge_count = n*(n-1)//2
    assert result["A"].shape == (4*n, 2*edge_count)
    assert result["Z"].shape == (4*n, n)
    exponent = np.array([.5*(amplitude[0, i]+amplitude[0, j]) for i, j in f.edges])
    np.testing.assert_allclose(result["A_exp"], result["A_static"]*np.exp(exponent), atol=4e-17, rtol=2e-13)
    k, beta = np.linspace(.2, .8, edge_count), np.linspace(.01, .1, edge_count)
    np.testing.assert_allclose(result["A"]@np.r_[k-beta, beta], result["explanation"]@np.r_[k, beta], atol=1e-16)
    # Unknown coefficients and gates cannot supply hidden topology to the dictionary.
    f.k_edge[:] = 0.
    f.gates = StructuralGates.static_baseline_limit(np.zeros(edge_count), n)
    unchanged = core.conditional_dictionary(cache, f, amplitude, a_times=cache["t"])
    np.testing.assert_array_equal(result["A"], unchanged["A"])
    zero = core.conditional_dictionary(cache, f, np.zeros_like(amplitude), a_times=cache["t"])
    np.testing.assert_array_equal(zero["A_static"], zero["A_exp"])
    np.testing.assert_array_equal(zero["explanation"][:, edge_count:], 0.)


def test_weak_signs_memory_units_and_shifted_initial_time_reference():
    f, packet = functional(), observation()
    f.gates = StructuralGates(s_edge=np.ones(6), s_mod=np.zeros(6), s_mem=np.full(4, .4), s_coi=0.)
    cache = core.prepare_weak_history(packet, f)
    t = cache["t"]
    v, dv = core.weak_test_functions(t)
    weighted = v*core.boole_weights(t)
    sine = np.sin(cache["theta"]@f.incidence)-np.sin(f.eta_edge)
    expected_A = np.vstack([f.incidence*row for row in weighted@sine])/.1
    expected_b = ((dv*core.boole_weights(t))@(cache["omega"]*f.mass)
                  -weighted@(cache["omega"]*f.damping)
                  +weighted@(cache["p"]*f.gain_p*f.storage_p)
                  -weighted@(cache["rf"]*f.gates.s_mem*f.j_omega_mem*f.lambda_mem)).ravel()/.1
    np.testing.assert_allclose(cache["A_static"], expected_A, atol=0., rtol=0.)
    np.testing.assert_allclose(cache["b"], expected_b, atol=2e-17, rtol=2e-14)
    decay = f.lambda_mem/f.tau_mem_sync
    integrals = weighted@(np.exp(-(t-t[0])[:, None]*decay)*f.gates.s_mem*f.j_omega_mem*f.lambda_mem)
    np.testing.assert_allclose(cache["Z"], np.vstack([np.diag(row) for row in integrals]), atol=2e-17, rtol=1e-13)
    np.testing.assert_allclose(cache["inertial"], 0., atol=2e-15, rtol=0.)


def test_candidate_time_shape_finiteness_and_fixed_physics_cache_validation():
    f = functional()
    cache = core.prepare_weak_history(observation(), f)
    a = np.zeros((len(cache["t"]), 4))
    for value, times in ((a[:-1], cache["t"]), (a, cache["t"]+.001), (a+np.nan, cache["t"])):
        with pytest.raises(ValueError):
            core.conditional_dictionary(cache, f, value, a_times=times)
    f.gain_p[0] += .1
    with pytest.raises(ValueError, match="fixed physics changed"):
        core.conditional_dictionary(cache, f, a, a_times=cache["t"])


def test_time_varying_candidate_dictionary_matches_actual_partial_force():
    f = functional()
    cache = core.prepare_weak_history(observation(), f)
    t = cache["t"]
    a = np.sin(30*t[:, None]+np.arange(4)[None, :])*.05
    result = core.conditional_dictionary(cache, f, a, a_times=t)
    k = np.linspace(.2, .7, 6)
    beta = np.linspace(.01, .1, 6)
    f.k_edge = k
    f.gates = StructuralGates(s_edge=np.ones(6), s_mod=beta/k, s_mem=np.ones(4), s_coi=0.)
    partial_force = np.array([f.gradients(f.pack_state(theta, omega, p, amplitude, rf))[0]
                              for theta, omega, p, amplitude, rf in
                              zip(cache["theta"], cache["omega"], cache["p"], a, cache["rf"])])
    v, _ = core.weak_test_functions(t)
    expected = ((v*core.boole_weights(t))@partial_force).ravel()/.1
    np.testing.assert_allclose(result["A"]@np.r_[k-beta, beta], expected, atol=4e-17, rtol=1e-13)


def test_block_memory_projection_and_physical_start_recovery_for_arbitrary_columns():
    A = np.array([[0., 0.], [1., 0.], [0., 2.]])
    Z = np.array([[2.], [0.], [0.]])
    coefficients = np.array([.3, .7])
    physical_rho = np.array([.037, -.024])
    systems = [dict(A=A, Z=Z, b=A@coefficients+Z[:, 0]*(rho/.1)) for rho in physical_rho]
    combined_A, combined_Z, b = core.combine_systems(systems)
    projection = core.project_memory(combined_A, combined_Z, b)
    solved = core.solve_nnls(projection["X"], projection["y"])
    np.testing.assert_allclose(solved["coefficients"], coefficients, atol=2e-16, rtol=0.)
    np.testing.assert_allclose(core.recover_memory(combined_A, b, coefficients, projection), physical_rho, atol=1e-17, rtol=0.)
    np.testing.assert_allclose(projection["orthogonality_A"], 0., atol=1e-15)
    np.testing.assert_allclose(projection["orthogonality_b"], 0., atol=1e-15)
    empty = core.project_memory(A, np.empty((3, 0)), A@coefficients)
    assert empty["memory_rank"] == 0
    assert core.recover_memory(A, A@coefficients, coefficients, empty).shape == (0,)


def test_relative_kkt_small_scale_empty_algebra_and_failed_call_accounting(monkeypatch):
    X, y = np.array([[2e-6]]), np.array([2e-6])
    wrong = core.nnls_kkt(X, y, np.zeros(1))
    assert not wrong["qualified"] and wrong["relative_violation"] == 1.
    right = core.solve_nnls(X, y)
    assert right["status"] == "qualified" and right["actual_nnls_calls"] == right["nnls_completed"] == right["nnls_qualified"] == 1
    np.testing.assert_allclose(right["coefficients"], 1.)
    def fail(*args, **kwargs):
        assert kwargs == {"maxiter": 200}
        raise RuntimeError("mock failure")
    monkeypatch.setattr(core, "nnls", fail)
    empty = core.solve_nnls(np.empty((1, 0)), y)
    assert empty["actual_nnls_calls"] == 0 and empty["loss"] == float(y@y)
    failed = core.solve_nnls(X, y)
    assert failed["status"] == "solver_unresolved" and failed["actual_nnls_calls"] == 1
    assert failed["nnls_completed"] == failed["nnls_qualified"] == 0
    zero = core.nnls_kkt(np.zeros((2, 3)), np.ones(2), np.zeros(3))
    assert zero["trivial_optimum"] and zero["qualified"]
    with pytest.raises(FloatingPointError):
        core.nnls_kkt(np.array([[1e200]]), np.array([1e200]), np.ones(1))


@pytest.mark.parametrize("n", [2, 4, 10])
def test_arithmetic_angle_basis_is_stable_orthonormal_and_deterministic(n):
    Q = core.angle_basis(n)
    tolerance = 8*n*np.finfo(np.float64).eps
    np.testing.assert_allclose(Q.T@Q, np.eye(n-1), atol=tolerance, rtol=0.)
    np.testing.assert_allclose(np.ones(n)@Q, 0., atol=tolerance, rtol=0.)
    np.testing.assert_allclose(Q@Q.T, np.eye(n)-np.ones((n, n))/n, atol=tolerance, rtol=0.)
    np.testing.assert_array_equal(Q, core.angle_basis(n))


@pytest.mark.parametrize("n", [2, 10])
def test_mock_backward_history_and_mass_projected_residual(n):
    class ConstantDrift:
        n_nodes = n
        mass = np.linspace(1., 2., n)
        def split_state(self, state):
            assert state.shape == (5*n,)
            return tuple(np.split(state, 5))
        def rhs(self, state):
            return np.arange(5*n)/100.
    f = ConstantDrift()
    endpoint = np.linspace(.1, .2, 5*n)
    original = endpoint.copy()
    sampled = core.backward_history(f, endpoint)
    fine = core.backward_history(f, endpoint, .00125)
    dense = core.backward_history(f, endpoint, dense=True)
    expected = endpoint+np.linspace(-.1, 0., 11)[:, None]*f.rhs(endpoint)
    np.testing.assert_allclose(sampled, expected, atol=8e-16, rtol=0.)
    np.testing.assert_allclose(sampled, fine, atol=2e-15, rtol=0.)
    assert dense.shape == (41, 5*n)
    np.testing.assert_array_equal(sampled[-1], endpoint)
    np.testing.assert_array_equal(endpoint, original)
    packet = dict(theta_history=sampled[:, :n]+.7, omega_history=sampled[:, n:2*n],
                  history_times=np.linspace(-.1, 0., 11), p_end=endpoint[2*n:3*n], sigma=0.)
    np.testing.assert_allclose(core.history_residual(f, sampled, packet), 0., atol=3e-15, rtol=0.)
    changed = copy.deepcopy(packet)
    changed["omega_history"][2, 0] += .01
    residual = core.history_residual(f, sampled, changed).reshape(11, 2*n)
    np.testing.assert_allclose(residual[2, n], -.1, atol=2e-16, rtol=0.)
