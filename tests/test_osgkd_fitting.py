import numpy as np
import pytest
import torch

from kiki.models.osgkd import OSGKD, OSGKDConfig
from kiki.models.osgkd_fitting import IntegrationWork, event_objective, integrate, joint_loss, tensor_events


def model():
    torch.manual_seed(37)
    return OSGKD(OSGKDConfig("C1", p_init_bound=4, e_init_bound=4),
                 mass=[1., 2., 3.], damping=[.1, .2, .3], F=np.eye(3),
                 Sp=[.1, .2, .3], Se=[.4, .5, .6], omega_scale=[1., 1., 1.],
                 theta_ref=[.1, .2, -.1], edge_index=[[0, 0, 1], [1, 2, 2]])


def event(count=20, identity="one"):
    theta = np.array([.2, .3, -.2]) + np.arange(11)[:, None] * .001
    omega = np.full((11, 3), .1)
    targets = {"theta": theta[-1] + np.arange(count+1)[:, None] * .001,
               "omega": np.full((count+1, 3), .1),
               "tau_m": np.full((count+1, 3), .03), "tau_e": np.full((count+1, 3), .02)}
    return {"id": identity, "history": {"theta": theta, "omega": omega},
            "t": np.arange(count+1) * .01, "targets": targets,
            "masks": {k: np.ones_like(v, dtype=bool) for k, v in targets.items()}}


def test_checkpoint_recomputation_matches_uncheckpointed_gradients_and_cost():
    a, b = model(), model()
    events = tensor_events([event(), event(13, "short")], "cpu")
    wa, wb = IntegrationWork(), IntegrationWork()
    finite = []
    la = event_objective(a, events, 20, [.1, .2, .3], work=wa, path_finite=finite)
    lb = event_objective(b, events, 20, [.1, .2, .3], recompute=True, work=wb)
    torch.testing.assert_close(la, lb, rtol=0, atol=0)
    joint_loss(la).backward()
    joint_loss(lb).backward()
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa.grad, pb.grad, rtol=1e-12, atol=1e-12)
    assert wa.rhs == {"forward": 4 * 2 * (20+13), "recompute": 0}
    assert wb.rhs["forward"] == wa.rhs["forward"]
    assert 0 < wb.rhs["recompute"] <= wb.rhs["forward"]
    assert wa.encoder_events == wb.encoder_events == 4
    assert len(finite) == 2 and all(bool(value) for value in finite)


def test_missing_port_values_and_decision_point_labels_never_enter_loss():
    a = event()
    b = event()
    b["targets"]["tau_m"][0] = 1e200
    for e in (a, b):
        e["masks"]["tau_e"][:, 1] = False
    b["targets"]["tau_e"][:, 1] = np.nan
    rom = model()
    values = [event_objective(rom, tensor_events([e], "cpu"), 20, [.1, .2, .3]) for e in (a, b)]
    torch.testing.assert_close(*values, rtol=0, atol=0)
    joint_loss(values[1]).backward()
    assert all(torch.isfinite(p.grad).all() for p in rom.parameters() if p.grad is not None)


def test_grouped_actual_horizons_preserve_event_weights_and_no_future_padding():
    rom = model()
    events = tensor_events([event(), event(7, "short")], "cpu")
    together = event_objective(rom, events, 20, [.1, .2, .3])
    separate = torch.cat([event_objective(rom, [e], 20, [.1, .2, .3]) for e in events])
    torch.testing.assert_close(together, separate, rtol=1e-12, atol=1e-12)
    assert together[1, 2].item() == 0
    with pytest.raises(ValueError, match="no available future"):
        event_objective(rom, tensor_events([event(0)], "cpu"), 20, [.1, .2, .3])


def test_integrator_matches_deployment_and_rejects_invalid_targets():
    rom = model()
    e = tensor_events([event()], "cpu")[0]
    state = rom.initialize(e["history"]["theta"], e["history"]["omega"])
    torch.testing.assert_close(integrate(rom, state, 20),
                               rom.rollout(state, np.arange(21)*.01, max_step=.005),
                               rtol=1e-12, atol=1e-12)
    bad = event()
    bad["targets"]["omega"][1, 0] = np.nan
    with pytest.raises(ValueError, match="observed target is nonfinite"):
        tensor_events([bad], "cpu")
