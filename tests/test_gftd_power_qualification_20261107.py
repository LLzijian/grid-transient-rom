"""Post-execution coverage supplement for both frozen Stage 0 graphs."""
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments"), str(ROOT / "tests")]
from test_gftd_kuramoto_architecture import parent_fixture
from kiki.models.gftd_kuramoto import GFTDKuramotoModel
from kiki.physics.kuramoto_functional import StructuralGates


def comparison(actual, expected, atol, rtol):
    actual, expected = np.asarray(actual), np.asarray(expected)
    error = np.abs(actual - expected)
    return dict(
        max_absolute_error=float(np.max(error)),
        max_relative_error=float(np.max(error / np.maximum(np.abs(expected), np.finfo(float).tiny))),
        relative_error_definition="max(abs(actual-reference)/max(abs(reference), float64.tiny))",
        max_tolerance_ratio=float(np.max(error / (atol + rtol * np.abs(expected)))),
        atol=atol, rtol=rtol, passed=bool(np.allclose(actual, expected, atol=atol, rtol=rtol)),
    )


def test_both_frozen_graphs_independent_gradient_and_power():
    start = time.perf_counter()
    output = ROOT / "results/gftd_repair_20261107"
    fixture_path = output / "qualification_fixtures.json"
    fixtures = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert len(fixtures) == 2
    source_paths = [
        fixture_path, Path(__file__), ROOT / "tests/test_gftd_kuramoto_architecture.py",
        ROOT / "src/kiki/models/gftd_kuramoto.py",
        ROOT / "src/kiki/physics/kuramoto_functional.py",
        ROOT / "experiments/r_memory_model_20261025.py",
    ]
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in source_paths}
    inputs, results = [], []
    for fixture in fixtures:
        parent = parent_fixture(fixture["chord"])
        assert parent.config() == fixture["config"], "fixture helper changed since qualification"
        np.testing.assert_allclose(float(parent.kappa), fixture["kappa"], atol=1e-15, rtol=0)
        np.testing.assert_allclose(float(parent.lambda_), fixture["lambda"], atol=1e-15, rtol=0)
        model = GFTDKuramotoModel.from_parent(parent)
        f = model.functional
        f.gamma_amp[:] = .5
        f.lambda_amp[:] = 1.
        f.tau_amp[:] = 5.
        states = np.asarray(fixture["parent_states"], dtype=np.float64)
        assert states.shape == (8, 40)
        amplitudes = np.random.default_rng(1107).uniform(-.1, .1, (8, 10))
        for modulation in (0., .5):
            f.gates = StructuralGates(f.gates.s_edge, f.gates.s_edge * modulation, np.ones(10))
            parameters = {field.name: getattr(f, field.name) for field in fields(f) if field.name != "gates"}
            parameters["gates"] = {field.name: getattr(f.gates, field.name) for field in fields(f.gates)}
            inputs.append(dict(chord=fixture["chord"], modulation=modulation,
                               parameters=parameters, parent_states=states, amplitudes=amplitudes,
                               delta_star=model.delta_star))
            for index, (old, amp) in enumerate(zip(states, amplitudes)):
                delta, omega, p, r = np.split(old, 4)
                state = f.pack_state(delta + model.delta_star, omega, p, amp, r)
                x = torch.tensor(state, dtype=torch.float64, requires_grad=True)
                theta_t, w_t, p_t, a_t, r_t = x.split(10)
                B = torch.tensor(f.incidence)
                eta = torch.tensor(f.eta_edge)
                phase = B.T @ theta_t
                h = B.abs().T @ (torch.tensor(f.gamma_amp) * a_t)
                gate = torch.tensor(f.gates.s_mod)
                kernel = 1 - gate + gate * h.exp()
                potential = (torch.tensor(f.gates.s_edge * f.k_edge) * kernel *
                             (eta.cos() - phase.cos() - eta.sin() * (phase - eta))).sum()
                energy = potential + .5 * (
                    (torch.tensor(f.mass) * w_t**2).sum()
                    + (torch.tensor(f.storage_p) * p_t**2).sum()
                    + (torch.tensor(f.lambda_amp) * a_t**2).sum()
                    + (torch.tensor(f.lambda_mem) * r_t**2).sum())
                gradient = torch.autograd.grad(energy, x)[0].detach().numpy()
                finite_difference = np.empty(50)
                for coordinate in range(50):
                    shift = np.eye(50)[coordinate] * 1e-5
                    finite_difference[coordinate] = (
                        f.hamiltonian(state + shift) - f.hamiltonian(state - shift)) / 2e-5
                rhs = f.rhs(state)
                chain_power = float(gradient @ rhs)
                independent_power = -float(
                    np.sum(f.damping * omega**2)
                    + np.sum(f.rate_p * (f.storage_p * p)**2)
                    + .2 * np.sum(gradient[30:40]**2)
                    + fixture["lambda"] / fixture["kappa"] * np.sum(r**2))
                checks = dict(
                    energy=comparison(f.hamiltonian(state), float(energy.detach()), 1e-11, 1e-10),
                    gradient=comparison(f.pack_state(*f.gradients(state)), gradient, 1e-11, 1e-10),
                    finite_difference=comparison(finite_difference, gradient, 1e-7, 1e-6),
                    actual_rhs_power=comparison(chain_power, independent_power, 1e-11, 1e-10),
                    audit_power=comparison(f.power_balance_audit(state)["expected_dH_dt"],
                                           independent_power, 1e-11, 1e-10))
                results.append(dict(chord=fixture["chord"], state_index=index,
                                    modulation=modulation, state=state, rhs=rhs,
                                    autograd_gradient=gradient, finite_difference_gradient=finite_difference,
                                    chain_power=chain_power, independent_power=independent_power,
                                    checks=checks, passed=all(c["passed"] for c in checks.values())))
    def serial(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(type(value).__name__)
    input_path = output / "power_supplement_inputs.json"
    input_path.write_text(json.dumps(dict(
        purpose="post-execution coverage supplement", seed=1107,
        amplitude_sampling="reset default_rng(1107) for each graph; reuse amplitudes for both gates",
        finite_difference_step=1e-5, fixtures=inputs), default=serial, indent=2, allow_nan=False),
        encoding="utf-8")
    hashes[str(input_path.relative_to(ROOT))] = hashlib.sha256(input_path.read_bytes()).hexdigest()
    receipt = dict(purpose="post-execution coverage supplement", utc=datetime.now(timezone.utc).isoformat(),
                   graph_count=2, parent_state_count=16, state_gate_checks=len(results),
                   status="PASSED" if all(row["passed"] for row in results) else "FAILED",
                   failure_reasons=[dict(chord=row["chord"], state_index=row["state_index"],
                                         modulation=row["modulation"], checks=row["checks"])
                                    for row in results if not row["passed"]],
                   repair_count=0, prior_qualification_receipt_modified=False,
                   probe_release_modified=False, elapsed_seconds=time.perf_counter()-start,
                   source_sha256=hashes, results=results)
    (output / "power_supplement_receipt.json").write_text(
        json.dumps(receipt, default=serial, indent=2, allow_nan=False), encoding="utf-8")
    assert receipt["status"] == "PASSED", receipt["failure_reasons"]
