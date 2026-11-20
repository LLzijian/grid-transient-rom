import importlib
from collections import Counter
from pathlib import Path

import numpy as np


def test_simpson_weights_and_nested_quadratic(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/osgkd_fit297_20261113"))
    recheck = importlib.import_module("recheck_ports")
    t = np.arange(11) * .01
    factors = np.arange(1., 11.)
    mass = factors + 1
    tm = (2 + 80 * t**2)[:, None] * factors
    omega = (2 * t + (80 / 3) * t**3)[:, None] * factors / mass
    work = Counter()
    for interval, nested in ((slice(None), False), (slice(0, 9), True), (slice(2, 11), True)):
        summary, evidence = recheck.quadrature_window(
            t[interval], omega[interval], tm[interval], np.zeros_like(tm[interval]),
            mass, np.zeros_like(mass), nested=nested, work=work,
        )
        np.testing.assert_allclose(evidence["S01"], evidence["impulse"], atol=1e-14, rtol=0)
        assert summary["s01_absolute_pass"]
        if nested:
            assert summary["nested_order_pass"] and summary["simpson_difference_pass"]
        else:
            assert summary["s01_reduction_pass"]
    assert work["history_integral_vectors"] == 13


def test_small_simpson_residual_does_not_pass_vector_order(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/osgkd_fit297_20261113"))
    recheck = importlib.import_module("recheck_ports")
    t = np.arange(9) * .01
    signs = np.array([1., -1.] * 5)
    tm = 1 + .05 * (-1.)**np.arange(9)[:, None] * signs
    omega = t[:, None] * (1 - (.05 / 3) * signs)
    summary, evidence = recheck.quadrature_window(
        t, omega, tm, np.zeros_like(tm), np.ones(10), np.zeros(10), nested=True, work=Counter(),
    )
    np.testing.assert_allclose(evidence["S01"], evidence["impulse"], atol=1e-14, rtol=0)
    assert summary["s01_absolute_pass"]
    assert not summary["nested_order_pass"]
    assert not summary["simpson_difference_pass"]
