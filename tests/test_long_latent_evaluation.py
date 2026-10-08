"""Synthetic contracts for the long evaluator; never opens private data."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest

pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
spec = importlib.util.spec_from_file_location("long_eval", ROOT / "experiments" / "evaluate_long_latent.py")
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)

def _event(n=112):
    return {"id":"synthetic", "group":"g", "label":"stable", "x":np.zeros((n,30)), "z":np.zeros((n,20)), "load":np.zeros(n)}

def test_100ms_alignment_and_metric_excludes_initial_sample():
    e = _event(); e["x"][10,0] = 2
    pred = np.zeros((101,33)); pred[0,0] = 2
    assert ev._metric_row(pred,e,1,np.ones(10))["normalized_sse"] == 0
    assert ev._metric_row(pred,e,1,np.ones(10))["n"] == 1000
    pred[1,0] = 1
    assert ev._metric_row(pred,e,1,np.ones(10))["normalized_sse"] > 0

def test_fine_and_coarse_grid_contract():
    assert ev.DT == .01
    assert np.array_equal(np.arange(1601)[::2]*.005, np.arange(801)*.01)

def test_short_event_has_prefix_not_complete_score():
    e = _event(50); pred = np.zeros((801,34))
    assert ev._metric_row(pred,e,1,np.ones(10)) is None
    row = ev._metric_row(pred,e,"prefix",np.ones(10))
    assert row["scope"] == "available_prefix" and row["n"] == 390

def test_causal_encoder_windows_do_not_require_future_beyond_available_prefix():
    e = _event(13); e["x"][:] = np.arange(13)[:,None]
    windows = np.stack([e["x"][i:i+11] for i in range(2)])
    assert windows.shape == (2,11,30)
    assert windows[1,-1,0] == 11

def test_generic_latent_dimension_does_not_affect_physical_scoring():
    assert ev._metric_row(np.zeros((2,34)),_event(12),"prefix",np.ones(10))["n"] == 10

def test_metric_uses_mass_weighted_coi_and_unavailable_horizon_is_none():
    e = _event(); pred = np.zeros((101,34)); pred[1,0] = 1; pred[1,1] = -1
    # Equal and opposite values do not have zero COI residual under unequal mass.
    assert ev._metric_row(pred,e,1,np.arange(1,11))["normalized_sse"] > 0
    assert ev._metric_row(np.zeros((101,34)),_event(110),1,np.ones(10)) is None

def test_numerical_convergence_window_includes_initial_sample():
    fine = np.zeros((101,20)); coarse = np.zeros((101,20)); fine[0,0] = 1
    de = ev.coi_project(fine[:,:10]-coarse[:,:10],np.ones(10))
    # The prescribed numerical mean includes the common t=0 point: denominator 101.
    assert np.isclose(np.mean(de*de), .09/101)

def test_q_short_prefix_is_clamped_before_window_construction():
    import torch
    class Encoder:
        def encode(self, windows):
            return torch.zeros((len(windows), 1), dtype=torch.float64)
    # 13 raw points allow windows ending at indices 10, 11, and 12.
    e = _event(13); autonomous = np.zeros((801, 31))
    row = ev._q_discrepancy(Encoder(), autonomous, e, 8)
    assert row["points"] == 3 and row["rms"] == 0

def test_teacher_aq_uses_complete_rhs_not_sparse_anchor():
    import torch
    class Teacher:
        rank = 3
        xi = torch.zeros((76,3), dtype=torch.float64)
        support = torch.ones((76,3), dtype=torch.float64)
        def latent_jacobian_at_reference(self):
            return torch.eye(3, dtype=torch.float64) * 7
    model = Teacher(); model.xi[1:4] = torch.eye(3, dtype=torch.float64)
    aq, anchor, source = ev._reported_aq(model, "teacher")
    assert np.array_equal(anchor, np.eye(3))
    assert np.array_equal(aq, np.eye(3)*7)
    assert source == "complete_teacher_rhs_reference_derivative"

def test_sparse_aq_retains_original_masked_xi_behavior():
    import torch
    class Sparse:
        rank = 3
        xi = torch.zeros((76,3), dtype=torch.float64)
        support = torch.ones((76,3), dtype=torch.float64)
    model = Sparse(); model.xi[1:4] = torch.eye(3, dtype=torch.float64)
    aq, anchor, source = ev._reported_aq(model, "challenger")
    assert np.array_equal(aq, anchor) and source == "masked_sparse_xi"

def test_sparse_aq_rank4_uses_all_four_latent_columns():
    import torch
    class Sparse:
        rank = 4
        xi = torch.zeros((77,4), dtype=torch.float64)
        support = torch.ones((77,4), dtype=torch.float64)
    model = Sparse(); model.xi[1:5] = torch.eye(4, dtype=torch.float64)
    aq, anchor, source = ev._reported_aq(model, "challenger")
    assert anchor.shape == (4,4)
    assert np.array_equal(anchor, np.eye(4))
    assert np.array_equal(aq, anchor) and source == "masked_sparse_xi"

def test_actual_teacher_constructor_loads_sparse_anchor_state_with_only_nn_keys_missing():
    import torch
    from kiki.models.private_pm import MechanicalROM
    from kiki.physics import complete_incidence
    from train_joint_latent import JointROM
    from train_teacher_latent import TeacherROM
    incidence = complete_incidence(10)[0]
    base = MechanicalROM(np.linspace(1,2,10), incidence, np.linspace(0,.2,10),
                         np.ones(10), np.r_[np.ones(26),np.zeros(19)], np.ones(10)*.1,
                         np.ones(10)*2, np.ones(10)*.3)
    anchor = JointROM(base, np.eye(10), np.ones(20), np.ones(72), 1., 1., latent_dim=3)
    teacher = TeacherROM(base, np.eye(10), np.ones(20), np.ones(72), 1., 1., latent_dim=3)
    missing, unexpected = teacher.load_state_dict(anchor.state_dict(), strict=False)
    assert set(missing) == {"W1.weight", "W1.bias", "W2.weight"}
    assert not unexpected and teacher.latent_jacobian_at_reference().shape == (3,3)
