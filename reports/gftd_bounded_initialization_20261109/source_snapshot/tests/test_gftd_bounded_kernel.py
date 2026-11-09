"""Bounded-kernel metadata/mock tests and separately metered fixed-point tests."""

from collections import OrderedDict
import copy
from dataclasses import fields, replace
import json

import numpy as np
import pytest

from kiki.models import GFTDKuramotoModel, gftd_discovery, gftd_fit297, gftd_fitting, gftd_kuramoto
from kiki.numerics import dop853_step
from kiki.physics.kuramoto_functional import (
    BOUNDED_MODULATION_KERNEL,
    EXP_MODULATION_KERNEL,
    GeneratingFunctionalKuramoto,
    StructuralGates,
)
from kiki.physics.graph import complete_incidence


def fixed_dp8_config(field):
    request = dict(
        format=gftd_fit297.FIT_FORMAT,
        protocol_sha256=gftd_fit297.PROTOCOL_SHA256,
        adaptation_indices=[0, 1],
        input_scale_delta=[.2, .3, .4],
        input_scale_omega=[.9, 1.1, 1.3],
        rstar=[.23, .33, .53],
        k_ref=1.,
        source_sha256={"artificial": "a" * 64},
    )
    parent = gftd_fit297.make_config(field, "modulated", request, 2, "fixed-artificial-parameters")
    return gftd_fitting.make_fixed_dp8_config(field, parent)


def test_metadata_legacy_checkpoint_and_kernel_aware_fit_cache(tmp_path, monkeypatch):
    model = GFTDKuramotoModel(n_nodes=3)
    legacy = model.functional
    bounded = replace(legacy, modulation_kernel=BOUNDED_MODULATION_KERNEL)
    assert legacy.modulation_kernel == EXP_MODULATION_KERNEL
    legacy_physics = gftd_fitting._physics(legacy)
    assert "modulation_kernel" not in legacy_physics
    assert gftd_fitting._plain(gftd_fitting._physics(bounded)) == {
        **gftd_fitting._plain(legacy_physics), "modulation_kernel": BOUNDED_MODULATION_KERNEL,
    }
    assert gftd_fitting._config(legacy, "modulated", 0.)["model_sha256"] == (
        "f14d37c896c979838766c3d6872a2cd29f5a68acfb6d3d74366b4e6057e2dccd"
    )
    assert gftd_fitting._config(legacy, "modulated", 0.)["model_sha256"] != (
        gftd_fitting._config(bounded, "modulated", 0.)["model_sha256"]
    )
    monkeypatch.setattr(gftd_fitting, "_BATCH_CACHE", OrderedDict())
    monkeypatch.setattr(gftd_fitting, "_build_preparation", lambda *args: {})
    _, old_hit, old_key = gftd_fitting._prepare_batch(legacy, [])
    _, bounded_hit, bounded_key = gftd_fitting._prepare_batch(bounded, [])
    assert not old_hit and not bounded_hit and old_key != bounded_key
    assert gftd_fit297._key(legacy, [], {}) != gftd_fit297._key(bounded, [], {})

    path = tmp_path / "legacy.json"
    model.save(path)
    saved = json.loads(path.read_bytes())
    assert saved["format"] == "gftd_kuramoto_5n_v2"
    assert dict(saved["functional"], gates=saved["gates"]) == gftd_fitting._plain(legacy_physics)
    for checkpoint_format in ("gftd_kuramoto_5n_v1", "gftd_kuramoto_5n_v2"):
        path.write_text(json.dumps(dict(saved, format=checkpoint_format)), encoding="utf-8")
        loaded = GFTDKuramotoModel.load(path).functional
        assert loaded.modulation_kernel == EXP_MODULATION_KERNEL
        assert gftd_fitting._hash(gftd_fitting._physics(loaded)) == gftd_fitting._hash(legacy_physics)


def test_metadata_v3_requires_marker_and_rejects_downgrade(tmp_path):
    with pytest.raises(ValueError, match="unknown amplitude modulation kernel"):
        GFTDKuramotoModel(n_nodes=3, modulation_kernel="unknown")
    model = GFTDKuramotoModel(n_nodes=3, modulation_kernel=BOUNDED_MODULATION_KERNEL)
    model.initialization_mode = "causal_two_stage"
    model.inference_config = fixed_dp8_config(model.functional)
    path = tmp_path / "bounded.json"
    model.save(path)
    saved = json.loads(path.read_bytes())
    assert saved["format"] == "gftd_kuramoto_5n_v3"
    assert saved["functional"]["modulation_kernel"] == BOUNDED_MODULATION_KERNEL
    loaded = GFTDKuramotoModel.load(path)
    assert loaded.functional.modulation_kernel == BOUNDED_MODULATION_KERNEL
    assert loaded.inference_config == model.inference_config
    assert loaded._uses_fixed_dp8

    unmarked = saved["functional"].copy()
    unmarked.pop("modulation_kernel")
    invalid = (
        (dict(saved, functional=unmarked), ValueError),
        (dict(saved, functional=dict(unmarked, modulation_kernel="unknown")), ValueError),
        (dict(saved, format="gftd_kuramoto_5n_v2"), ValueError),
        (dict(saved, format="gftd_kuramoto_5n_v2", functional=unmarked), gftd_fitting.InitializationFailure),
    )
    for payload, error in invalid:
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(error):
            GFTDKuramotoModel.load(path)

    class UnversionedFunctional(GeneratingFunctionalKuramoto):
        pass

    model.functional = UnversionedFunctional(
        **{field.name: getattr(model.functional, field.name) for field in fields(model.functional)}
    )
    with pytest.raises(ValueError, match="functional subclass"):
        model.save(path)


def test_mock_public_initialize_and_rollout_preserve_dp8_and_fit297_scales(tmp_path, monkeypatch):
    model = GFTDKuramotoModel(n_nodes=3, modulation_kernel=BOUNDED_MODULATION_KERNEL)
    model.initialization_mode = "causal_two_stage"
    model.inference_config = fixed_dp8_config(model.functional)
    checkpoint = tmp_path / "bounded.json"
    model.save(checkpoint)
    model = GFTDKuramotoModel.load(checkpoint)
    field = model.functional
    config = model.inference_config["fit297_config"]
    packet = dict(theta_history=np.zeros((11, 3)), omega_history=np.zeros((11, 3)),
                  history_times=np.linspace(-.1, 0., 11), p0=np.array([.02, -.01, .03]))
    solves, gates, future = [], [], []

    def infer(f, data, *, mode, role, call_limit, config, stepper, initial=None, feasible=None):
        assert f is field and f.modulation_kernel == BOUNDED_MODULATION_KERNEL and stepper is dop853_step
        assert set(data) == {"theta_history", "omega_history", "history_times", "p_end", "sigma"}
        assert config == model.inference_config["fit297_config"]
        simple = role == "simple"
        solves.append((role, call_limit))
        u = np.array([.2, -.1, .3]) if simple else initial.copy()
        if not simple:
            np.testing.assert_array_equal(initial, feasible)
            np.testing.assert_array_equal(initial, [0., 0., 0., .2, -.1, .3])
            u[0] = .5
        state = gftd_fit297.endpoint(data, f, u, mode, role, config)
        np.testing.assert_array_equal(state[6:9], packet["p0"])
        np.testing.assert_array_equal(state[-3:], np.asarray(config["r_scale"]) * u[-3:])
        return dict(best=dict(u=u, state=state, observation_sse=2. if simple else 1.), status="converged",
                    counts=gftd_fitting._counts(stepper), same_step_feasible_sse=2., projected_gradient_kkt=0.)

    def integrate(f, state, step, duration, counts, *, backward=False, dense=False, stepper=None):
        assert f is field and stepper is dop853_step and dense is False
        gates.append((step, duration, backward))
        return np.repeat(state[None, :], int(round(duration / .01)) + 1, axis=0)

    def rollout_step(rhs, state, step):
        assert rhs == field.rhs
        future.append(step)
        return state.copy()

    monkeypatch.setattr(gftd_fitting, "infer_endpoint", infer)
    monkeypatch.setattr(gftd_fitting, "_integrate", integrate)
    monkeypatch.setattr(gftd_kuramoto, "dop853_step", rollout_step)
    with pytest.raises(ValueError, match="requires exactly"):
        model.initialize(dict(packet, event_id="forbidden"))
    path = model.rollout_state(packet, np.array([0., .005, .01]))
    assert solves == [("simple", 600), ("smooth", 1200)]
    assert gates == [(.0025, .1, True), (.0025, .1, False), (.00125, .1, True), (.00125, .1, False),
                     (.01, 1., False), (.005, 1., False)] * 2
    assert future == [.005, .005]
    assert model.last_initialization["method"] == "fixed_dp8"
    assert model.last_initialization["returned_point"] == "stage2"
    assert model.last_initialization["final_numerical"]["passed"] is True
    assert config["history_theta_scale"] == [.2, .3, .4] and config["history_omega_scale"] == [.9, 1.1, 1.3]
    np.testing.assert_array_equal(path[0, 6:9], packet["p0"])
    broken_state = path[0].copy()
    broken_state[6] += .1
    monkeypatch.setattr(gftd_fitting, "_initialize_validated", lambda *args, **kwargs: (broken_state, {}))
    with pytest.raises(gftd_fitting.InitializationFailure, match="hard constraint"):
        model.initialize(packet)
    print(f"mock_endpoint_solves={len(solves)} mock_gate_integrations={len(gates)} mock_rollout_steps={len(future)}")


def fixed_artificial_cases():
    incidence, _ = complete_incidence(3)
    one, zero = np.ones(3), np.zeros(3)
    theta_star = np.array([.1, -.05, 0.])
    constants = dict(mass=[1., 1.5, 2.], damping=.2 * one, k_edge=[1., .8, 1.2],
                     eta_edge=incidence.T @ theta_star, gamma_amp=[.5, -.25, .75],
                     lambda_amp=one, lambda_mem=one, tau_amp=one, tau_mem_sync=one, tau_mem_coi=1.,
                     j_omega_mem=one, storage_p=one, gain_p=one, rate_p=one)
    i = np.r_[theta_star, zero, zero, zero, zero]
    ii = np.array([8., 0., 0., .1, -.1, .05, .2, 0., -.1, .1, -.2, .3, .01, -.02, .03])
    iii, iv = ii.copy(), ii.copy()
    iii[9:12], iv[9:12] = [2000., 0., 0.], [-2000., 0., 0.]
    cases = {
        "A": (StructuralGates(one, zero, one), (i, ii)),
        "B": (StructuralGates([1., 1., 0.], [.5, 0., 0.], one), (i, ii)),
        "C": (StructuralGates(one, [.5, 0., 1.], one), (i, ii, iii, iv)),
    }
    return constants, cases


def sample(field, state, monkeypatch):
    h = field.hamiltonian(state)
    couplings = field.effective_edge_couplings(field.split_state(state)[3])
    gradients, calculate = [], field.gradients

    def capture(value):
        result = calculate(value)
        gradients.append(result)
        return result

    with monkeypatch.context() as local:
        local.setattr(field, "gradients", capture)
        rhs = field.rhs(state)
    assert len(gradients) == 1
    g = field.pack_state(*gradients[0])
    assert np.isfinite(np.r_[h, couplings, g, rhs]).all()
    ra, rm = field.dissipative_operators(state)
    _, gw, gp, ga, gr = gradients[0]
    power = float(g @ rhs)
    dissipation = float(np.sum(field.damping * (gw / field.mass)**2 + field.rate_p * gp**2 + ra * ga**2)
                        + gr @ rm @ gr)
    assert abs(power + dissipation) <= 1e-11 * max(1., abs(power), dissipation)
    return h, couplings, g, rhs


def assert_same_sample(left, right):
    for value, expected in zip(left, right, strict=True):
        np.testing.assert_allclose(value, expected, atol=1e-12, rtol=1e-11)


def test_artificial_fixed_points_storage_power_and_versioned_clones(tmp_path, monkeypatch):
    constants, cases = fixed_artificial_cases()
    for name, (gates, states) in cases.items():
        field = GeneratingFunctionalKuramoto(**constants, gates=gates, modulation_kernel=BOUNDED_MODULATION_KERNEL)
        for index, state in enumerate(states):
            result = sample(field, state, monkeypatch)
            if index < 2:
                for step in (1e-5, 5e-6):
                    differences = [(field.hamiltonian(state + delta) - field.hamiltonian(state - delta)) / (2. * step)
                                   for delta in step * np.eye(state.size)]
                    np.testing.assert_allclose(differences, result[2], atol=1e-7, rtol=1e-6)
            else:
                expected = .5 + .5 * (np.e if index == 2 else 1. / np.e)
                np.testing.assert_allclose(result[1][0], expected, atol=0., rtol=1e-15)
            if name == "A":
                original = GeneratingFunctionalKuramoto(**constants, gates=gates)
                assert_same_sample(result, sample(original, state, monkeypatch))
            if name == "C" and index == 1:
                model = GFTDKuramotoModel(n_nodes=3, mass=field.mass, damping=field.damping)
                model.functional, model.gates = field, field.gates
                path = tmp_path / "bounded.json"
                model.save(path)
                restored = GFTDKuramotoModel.load(path).functional
                cloned = gftd_fitting._parameters(field, field.k_edge, field.k_edge * field.gates.s_mod)
                canonical, transformed = field.canonical_gauge_transform(state)
                np.testing.assert_array_equal(transformed, state)
                for equivalent in (restored, copy.deepcopy(field), cloned, canonical):
                    assert equivalent.modulation_kernel == BOUNDED_MODULATION_KERNEL
                    assert_same_sample(sample(equivalent, state, monkeypatch), result)


def test_artificial_conditional_dictionary_matches_bounded_force_with_gate_independence():
    constants, cases = fixed_artificial_cases()
    field = GeneratingFunctionalKuramoto(
        **constants, gates=cases["C"][0], modulation_kernel=BOUNDED_MODULATION_KERNEL,
    )
    changed = replace(field, k_edge=[.7, 0., 1.1], gates=cases["B"][0])
    ports, _, _ = gftd_discovery._port_constants(field)
    fixed = dict(ports, mass=field.mass, damping=field.damping, eta_edge=field.eta_edge, gamma_amp=field.gamma_amp)
    for step in gftd_discovery.HISTORY_STEPS:
        t = np.linspace(-.1, 0., int(round(.1 / step)) + 1)
        theta = np.array([8., 0., 0.]) + (t + .1)[:, None] * np.array([.5, -.25, .125])
        force = np.sin(theta @ field.incidence) - np.sin(field.eta_edge)
        v, _ = gftd_discovery.weak_test_functions(t)
        weighted = v * gftd_discovery.boole_weights(t)
        columns = np.vstack([field.incidence * row for row in weighted @ force]) / gftd_discovery.STATE_SCALE
        cache = dict(t=t, theta=theta, A_static=columns, Z=np.zeros_like(columns), b=np.zeros(len(columns)),
                     fixed_parameters=fixed)
        for a in (np.zeros_like(theta), (t + .1)[:, None] * np.array([1., -.5, .25])):
            systems = [gftd_discovery.conditional_dictionary(cache, f, a, a_times=t) for f in (field, changed)]
            np.testing.assert_array_equal(systems[0]["A_exp"], systems[1]["A_exp"])
            if not np.any(a):
                np.testing.assert_allclose(systems[0]["A_exp"], columns, atol=1e-11, rtol=1e-10)
            else:
                assert not np.array_equal(systems[0]["A_exp"], columns)
            for f, system in zip((field, changed), systems, strict=True):
                c = f.gates.s_edge * f.k_edge
                beta = c * f.gates.s_mod
                couplings = np.array([f.effective_edge_couplings(row) for row in a])
                weak_force = (weighted @ ((couplings * force) @ f.incidence.T)).ravel() / gftd_discovery.STATE_SCALE
                np.testing.assert_allclose(system["A"] @ np.r_[c - beta, beta], weak_force, atol=1e-11, rtol=1e-10)
