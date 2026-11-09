from dataclasses import dataclass
from fractions import Fraction as F
from types import SimpleNamespace

import pytest

from kiki.evaluation import gftd_analysis as analysis
from kiki.physics.kuramoto_functional import BOUNDED_MODULATION_KERNEL, EXP_MODULATION_KERNEL


@dataclass
class _Metadata:
    modulation_kernel: str
    k_edge: tuple
    gates: SimpleNamespace
    eta_edge: tuple
    damping: tuple
    n_nodes = 3
    edges = ((0, 1), (0, 2), (1, 2))
    gamma_amp = (1, 0, 0)
    mass = storage_p = lambda_amp = lambda_mem = (1, 1, 1)
    rate_p = tau_amp = tau_mem_sync = (1, 1, 1)
    j_omega_mem = gain_p = (1, 1, 1)
    gate_weight_amp = gate_weight_mem = None


def _metadata(kernel=BOUNDED_MODULATION_KERNEL, *, c=(1, 1, 0),
              beta=(F(1, 2), 0, 0), delta=0, damping=(1, 1, 1)):
    gates = SimpleNamespace(s_edge=(1, 1, 1), s_mod=tuple(map(str, beta)),
                            s_mem=(0, 0, 0), s_coi=0)
    return _Metadata(kernel, c, gates, (str(delta), 0, 0), damping)


def test_outward_quantization_and_signed_accumulation():
    grid = 2**128
    for value in (F(0), F(1, 3), -F(1, 3), F(1, 2**129), -F(1, 2**129)):
        lo, hi = analysis.outward_interval((value, value))
        assert lo <= value <= hi
        assert (lo*grid).denominator == (hi*grid).denominator == 1
        assert hi-lo <= F(1, grid)
    assert analysis.outward_interval((-F(1, 2**129),)*2) == (-F(1, grid), 0)

    total = F(0)
    bounds = (total, total)
    for value in (F(1, 3), -F(1, 3), F(1, 2**129), -F(1, 2**129), F(0), F(0))*3:
        magnitude = abs(value)
        term = analysis._scale(analysis.outward_interval((magnitude, magnitude)), -1 if value < 0 else 1)
        bounds = analysis.outward_interval(analysis._add(bounds, term))
        total += value
        assert bounds[0] <= total <= bounds[1]


def test_fixed_bounded_kernel_intervals_and_domains():
    budget = analysis.TaylorBudget(12)
    for h in (-F(1, 4), F(0), F(1, 4)):
        exp_lo, exp_hi = budget.interval(2*h, "exp")
        assert 0 < exp_lo <= exp_hi
        assert -1 < analysis._tanh_interval((exp_lo, exp_hi))[0]
        lo, hi = analysis._bounded_kernel_interval(h, budget.interval)
        assert 1-abs(h) <= lo <= hi <= 1/(1-abs(h))
        if h < 0:
            assert hi < 1
        elif h > 0:
            assert lo > 1
        else:
            assert lo == hi == 1
    assert budget.calls == 12

    with pytest.raises(ValueError, match="strictly positive"):
        analysis._tanh_interval((-1, 1))
    with pytest.raises(ValueError, match="out of order"):
        analysis._tanh_interval((1, -1))
    rejected = analysis.TaylorBudget(1)
    with pytest.raises(ValueError, match="requires"):
        rejected.interval(1, "exp")
    with pytest.raises(ValueError, match="budget exhausted"):
        rejected.interval(0, "exp")
    assert rejected.calls == 1


def test_bounded_radius_grid_and_rounded_center():
    center = (0, 0, 0)
    cert = analysis.l2_certificate(_metadata(), center)
    assert cert["status"] == "certified"
    assert cert["taylor_calls"] == 0
    rows = cert["candidates"]
    # The angle cap (7/15) is tighter than the gamma cap (1/2).
    assert [F(row["R"]) for row in rows] == [F(7, 15)/2**j for j in range(13)]
    assert rows[0]["reason"] == "bounded_modulation_domain"
    assert rows[1]["status"] == "certified"
    assert F(rows[1]["b"]) == F(7, 30)

    offset = F(1, 1024)
    metadata = _metadata(delta=offset)
    cert = analysis.l2_certificate(metadata, center)
    row = cert["selected"]
    assert F(row["g"]) == F(3, 2)*offset + offset**2/4
    point = analysis.l2_point(cert, metadata, (0,)*15)
    assert point["status"] == "certified"
    assert point["taylor_calls"] == 9
    assert point["epsilon"] is None
    assert F(point["V_upper"]) == F(point["energy_interval"][1]) + F(row["g"])**2/(2*F(row["mu"]))


def test_empty_modulation_and_static_compatibility(monkeypatch):
    def unexpected_kernel(*args):
        pytest.fail("bounded beta=0 must skip the kernel")

    monkeypatch.setattr(analysis, "_bounded_kernel_interval", unexpected_kernel)
    center = (0, 0, 0)
    bounded = _metadata(beta=(0, 0, 0))
    static = _metadata(EXP_MODULATION_KERNEL, beta=(0, 0, 0))
    bounded_cert = analysis.l2_certificate(bounded, center)
    static_cert = analysis.l2_certificate(static, center)
    assert F(bounded_cert["candidates"][0]["b"]) == 0
    assert F(static_cert["candidates"][0]["h"]) == F(7, 15)
    budget = analysis.TaylorBudget(6+8)
    assert analysis.l2_point(bounded_cert, bounded, (0,)*15, taylor_budget=budget)["taylor_calls"] == 6
    result = analysis.l2_point(static_cert, static, (0,)*15, taylor_budget=budget)
    assert result["taylor_calls"] == 8
    assert result["energy_interval"] == ["0", "0"]
    assert budget.calls == 14


def test_l2_branch_failures_keep_counts():
    center = (0, 0, 0)
    for metadata, reason in (
        (_metadata(c=(1, 0, 0)), "active_graph_disconnected"),
        (_metadata(damping=(0, 1, 1)), "strict_storage_or_dissipation_missing"),
        (_metadata(EXP_MODULATION_KERNEL), "static_exp_requires_zero_beta"),
        (_metadata("unknown"), "unknown_kernel"),
    ):
        result = analysis.l2_certificate(metadata, center)
        assert result["status"] == "not_certified"
        assert result["reason"] == reason
        assert result["taylor_calls"] == 0

    metadata = _metadata()
    cert = analysis.l2_certificate(metadata, center)
    budget = analysis.TaylorBudget(2)
    point = analysis.l2_point(cert, metadata, (0,)*15, taylor_budget=budget)
    assert point["reason"] == "point_envelope_failed"
    assert "budget exhausted" in point["detail"]
    assert point["taylor_calls"] == budget.calls == 2
