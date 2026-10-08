"""Small synthetic contracts for the Loop0090 numerical solve helper."""
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("harmonic0090", ROOT / "experiments" / "harmonic_solve_0090.py")
mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)


def test_complex_multi_rhs_returns_known_solution_and_full_artifact():
    h = np.array([[2 + 1j, -1j], [3, 1 - 2j]], complex)
    x = np.array([[1 - 2j, 2], [3j, -1 + .5j]], complex); b = h @ x
    got = mod.solve_scaled(h, b, time.perf_counter() + 5)
    assert got["qa_pass"] and got["error"] is None
    assert np.allclose(got["X"], x, atol=1e-13, rtol=0)
    assert np.allclose(got["H"], h) and np.allclose(got["B"], b)
    assert got["singular_values"].shape == (2,)
    assert got["metrics"]["backward_original"] < 1e-13
    assert got["solve_attempts"] == got["svd_attempts"] == 1


def test_well_conditioned_hidden_similarity_preserves_reconstructed_solution():
    h = np.array([[3 + .2j, 1], [-.5j, 2 - .4j]], complex)
    b = np.array([[1 + 2j, -1], [.5, 3j]], complex)
    transform = np.array([[1.2, .1], [.2, .9]], complex)
    direct = mod.solve_scaled(h, b, time.perf_counter() + 5)
    changed = mod.solve_scaled(transform @ h @ np.linalg.inv(transform), transform @ b, time.perf_counter() + 5)
    assert direct["qa_pass"] and changed["qa_pass"]
    assert np.allclose(np.linalg.inv(transform) @ changed["X"], direct["X"], atol=1e-12, rtol=0)


def test_nonsingular_algebraic_feedthrough_block_is_not_rejected():
    # An algebraic-looking row is just a nonsingular complex block to this helper.
    h = np.array([[1, -2], [0, 3]], complex); b = np.array([[2], [6]], complex)
    got = mod.solve_scaled(h, b, time.perf_counter() + 5)
    assert got["qa_pass"] and np.allclose(h @ got["X"], b, atol=1e-13, rtol=0)


def test_singular_failure_preserves_scaled_inputs_without_retry():
    h = np.array([[1, 2], [2, 4]], complex); b = np.array([[1], [2]], complex)
    got = mod.solve_scaled(h, b, time.perf_counter() + 5)
    assert not got["qa_pass"] and got["error"].startswith("solve_")
    assert np.array_equal(got["H"], h) and np.array_equal(got["B"], b)
    assert got["equilibrated_H"] is not None and got["singular_values"] is not None


def test_expired_deadline_does_not_call_scipy_solve(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("solve called after an expired deadline")
    monkeypatch.setattr(mod.scipy.linalg, "solve", forbidden)
    got = mod.solve_scaled(np.eye(2, dtype=complex), np.ones((2, 1), complex), time.perf_counter() - 1)
    assert got["error"] == "deadline_before_scaling" and got["X"] is None
