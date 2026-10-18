import copy
import hashlib
from pathlib import Path
import numpy as np
import pytest
from experiments import closure_short_history_20261019 as short


BASE = {"postfault_horizon_s": 15., "disable_pss": False,
        "sample_interval_s": .01, "integration_step_s": .005,
        "andes_version": "2.0.0", "case": "unchanged", "custom": {"nested": [1]}}


def test_config_has_only_declared_changes():
    original = copy.deepcopy(BASE)
    config = short.build_short_config(BASE, step=.0025,
                                      stop_rule="first_record_plus_100ms")
    assert BASE == original
    changed = {key for key in set(BASE) | set(config) if BASE.get(key) != config.get(key)}
    assert changed == {"postfault_horizon_s", "integration_step_s", "short_history_stop_rule"}
    config["custom"]["nested"].append(2)
    assert BASE == original


@pytest.mark.parametrize("step", [.005, .0025])
def test_strict_stop_cannot_fake_full_history(step):
    clearing, delay = 1.12, step
    config = short.build_short_config(BASE, step=step)
    result = short.short_history_metadata(np.arange(10)*.01, clearing+delay,
                                          clearing, clearing+.1, config, "unknown")
    assert result["label"] == "unknown"
    assert result["history_eligible"] is False
    assert result["forecast_origin_since_clearing_s"] is None
    assert result["sample_absolute_times_s"][-1] <= clearing+.1
    assert result["sample_since_clearing_s"][0] == pytest.approx(delay)


@pytest.mark.parametrize("step", [.005, .0025])
def test_explicit_observed_record_endpoint(step):
    clearing = 1.12
    config = short.build_short_config(BASE, step=step, stop_rule="first_record_plus_100ms")
    result = short.short_history_metadata(np.arange(11)*.01, clearing+step,
                                          clearing, clearing+step+.1, config, "unknown")
    assert result["history_eligible"] is True
    assert result["forecast_origin_since_clearing_s"] == pytest.approx(.1+step)
    assert result["full_horizon_outcome"] == "unknown"
    assert result["continuation_supported"] is False


def test_no_stable_label_and_no_future_integration():
    config = short.build_short_config(BASE)
    with pytest.raises(ValueError, match="stable outcome"):
        short.short_history_metadata(np.arange(10)*.01, 1.005, 1., 1.1, config, "stable")
    with pytest.raises(ValueError, match="beyond"):
        short.short_history_metadata(np.arange(10)*.01, 1.005, 1., 1.11, config, "unknown")
    with pytest.raises(ValueError, match="did not reach"):
        short.short_history_metadata(np.arange(5)*.01, 1.005, 1., 1.05, config, "unknown")
    result = short.short_history_metadata(np.arange(5)*.01, 1.005, 1., 1.05, config, "unstable")
    assert result["full_horizon_outcome"] == "unstable"
    assert result["history_eligible"] is False


def test_cannot_extrapolate_or_hide_record_gaps():
    config = short.build_short_config(BASE)
    with pytest.raises(ValueError, match="extrapolated"):
        short.short_history_metadata(np.arange(11)*.01, 1.005, 1., 1.1, config, "unknown")
    with pytest.raises(ValueError, match="extrapolated"):
        short.short_history_metadata([0., .01, .03], 1.005, 1., 1.1, config, "unknown")


def test_train_endpoint_and_domain_gate():
    event = dict(bus=3, line="Line_3", location=0., duration_s=.125, xf=.00316227766)
    allowed = [dict(event, duration_s=.1, xf=.001)]
    assert short.validate_train_condition(event, allowed) == event
    for change in [dict(bus=4), dict(line="Line_99"), dict(location=1.),
                   dict(duration_s=float("nan")), dict(xf=.1)]:
        with pytest.raises(ValueError):
            short.validate_train_condition(dict(event, **change), allowed)


def test_fork_retains_original_source_identity():
    source = Path(short.__file__).with_name("generate_private_dataset.py")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == short.SOURCE_SIMULATOR_SHA256


@pytest.mark.parametrize("kwargs", [dict(step=.01), dict(stop_rule="truncate_full_run")])
def test_unsupported_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        short.build_short_config(BASE, **kwargs)


@pytest.mark.parametrize("rule,expected_count", [("clearing_plus_100ms", 10),
                                                ("first_record_plus_100ms", 11)])
def test_simulator_stops_integrator_using_observed_record_start(monkeypatch, rule, expected_count):
    # Fake integrator supplies timestamps/state storage only. This verifies the
    # actual simulator control flow; it is not a numerical FOM qualification.
    import sys
    from types import SimpleNamespace as NS
    def variable(values, address=None):
        return NS(v=np.asarray(values), a=np.arange(10) if address is None else address,
                  v_code="x")
    gen = NS(M=variable(np.ones(10)), omega=variable(np.ones(10)),
             delta=variable(np.zeros(10)))
    for name in ("tm", "Pe", "Qe", "te", "vd", "vq"):
        setattr(gen, name, variable(np.ones(10)))
    dae = NS(t=0., xy=np.ones(20), x=np.zeros(10), y=np.zeros(10),
             y_name=[str(i) for i in range(10)], f=np.zeros(10), g=np.zeros(10))
    dae.ts = NS(t=[], x=[], y=[])
    system = NS(models={}, GENROU=gen, dae=dae,
                Line=NS(idx=NS(v=["line"]), bus1=NS(v=[3]), bus2=NS(v=[4])),
                Fault=NS(uf=NS(v=np.zeros(1))), PFlow=NS(run=lambda: True),
                Bus=NS(v=variable(np.ones(10)), a=variable(np.zeros(10))))
    system.setup = lambda: None
    system.add = lambda *args: None
    config = short.build_short_config(dict(BASE, fault_start_s=1.,
                   newton_tolerance=1e-7, newton_max_iterations=30, fault_type="temporary_bus_three_phase_ground",
                   pre_fault_speed_tolerance_pu=1e-6, pre_fault_angle_drift_rad=1e-5,
                   los_hold_samples=5), stop_rule=rule)
    tds = NS(config=NS(), test_ok=True, busted=False, itm_step=lambda: True,
             init=lambda: None, do_switch=lambda: None)
    system.TDS = tds
    final_steps = []
    def run(**kwargs):
        step_index = 0
        while dae.t < tds.config.tf - 1e-10:
            step_index += 1
            dae.t = round(step_index * config["integration_step_s"], 10)
            system.Fault.uf.v[:] = float(1. < dae.t < 1.1)
            dae.x = np.ones(10) * dae.t
            dae.ts.t.append(dae.t)
            dae.ts.x.append(dae.x.copy())
            dae.ts.y.append(dae.y.copy())
            assert tds.itm_step()
            tds.do_switch()
            tds.check_criteria()
            final_steps.append(dae.t)
        return True
    tds.run = run
    fake = NS(andes=NS(__version__="2.0.0", load=lambda *a, **kw: system,
                      get_case=lambda name: name),
              physical_parameters=lambda ss: "same", state_mapping=lambda ss: [],
              _solver_load_kwargs=lambda d: ({}, None), check_arrays=lambda *a: None,
              los_index=lambda *a: None)
    monkeypatch.setitem(sys.modules, "experiments.generate_private_dataset", fake)
    diagnostics = {}
    arrays, metadata, *_ = short._simulate_short_impl(
        dict(line="line", bus=3, location=0., duration_s=.1, xf=.001),
        config, diagnostics=diagnostics)
    expected_end = 1.2 if rule == "clearing_plus_100ms" else 1.205
    assert max(final_steps) == pytest.approx(expected_end)
    assert len(arrays["t"]) == expected_count
    assert metadata["label"] == "unknown"
    assert metadata["history_eligible"] == (expected_count == 11)
    assert metadata["record_start_absolute_s"] == pytest.approx(1.105)
    assert diagnostics["solver_final_time_s"] == pytest.approx(expected_end)




def test_genuine_probe_float_endpoint_residue_does_not_create_tiny_step():
    # Measured ANDES 2.0.0 failure from pilot_0000: observed first record
    #1.0501s, requested1.1501s, last converged state1.150099999999998s.
    requested, converged = 1.1501, 1.150099999999998
    assert short._converged_endpoint_target(requested, converged) == converged
    # Do not hide an actual short step or relax the integration gate to1e-8.
    assert short._converged_endpoint_target(requested, requested-1e-10) == requested
    assert short._converged_endpoint_target(requested, requested+.001) == requested
    assert short._converged_endpoint_target(requested, requested) == requested
