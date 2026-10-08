"""Deterministic unit tests for the causal input contract.

These tests are deliberately pure: they exercise only dataclass/enum logic in
``kiki.data.contracts`` and never touch data, models, or results. They pin the
project's core discipline — no forbidden field may ever be an allowed model
input, and the documented causal clearing-state fields must always be allowed.
"""
import pytest

from kiki.data.contracts import InputContract, ModelDataRequest, PmMode, RequestMode


def test_forbidden_fields_never_also_allowed():
    ic = InputContract()
    allowed = set(ic.allowed_static) | set(ic.allowed_exogenous)
    leak = allowed & set(ic.forbidden)
    assert leak == set(), f"forbidden fields leaked into allowed inputs: {leak}"


def test_documented_causal_static_fields_allowed():
    ic = InputContract()
    causal = {"delta0", "omega0", "Pm_equilibrium", "Pm_clear", "mass", "incidence"}
    assert causal <= set(ic.allowed_static)


def test_observed_teacher_fields_are_only_exogenous_or_forbidden():
    ic = InputContract()
    assert "Pm" in ic.allowed_exogenous  # time-varying Pm is the causal input
    for f in ("Pe", "bus_v", "bus_a"):
        assert f in ic.forbidden, f"{f} must be forbidden by the causal contract"


def test_discipline_guard_fields_forbidden():
    ic = InputContract()
    for f in ("label", "outcome_label", "valid_length", "valid_mask", "fault_line"):
        assert f in ic.forbidden, f"{f} must be forbidden by the causal contract"


def test_prefix_steps_rejects_single_sample():
    with pytest.raises(ValueError):
        ModelDataRequest.prefix_steps(1)


def test_prefix_steps_two_samples_ok():
    req = ModelDataRequest.prefix_steps(2)
    assert req.n_steps == 2
    assert req.prospective is True
    assert req.mode is RequestMode.PREFIX_STEPS
    assert req.pm_mode is PmMode.ZOH
