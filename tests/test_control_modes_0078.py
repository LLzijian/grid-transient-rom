import importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("control_modes", ROOT / "experiments" / "check_control_modes_0078.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)


def test_block_gauge_has_expected_shape_and_orthonormality():
    q = mod.block_diag(mod.null_space(np.ones((1, 10))), np.eye(23))
    assert q.shape == (33, 32)
    assert np.max(np.abs(q.T @ q - np.eye(32))) < 1e-12


def test_finite_difference_helper_linear_map():
    a = np.arange(9, dtype=float).reshape(3, 3)
    x = np.array([.2, -.4, .7])
    got = mod.finite_difference(lambda y: a @ y, x, 1e-5)
    assert np.allclose(got, a, atol=1e-10, rtol=1e-10)


def test_autodiff_and_fd_use_same_origin_for_nonlinear_field():
    import torch
    class Toy:
        def field(self, x):
            # Invariant to common angle, nonlinear away from the origin.
            d = x[:10] - x[:10].mean()
            return torch.cat((d + d.square(), x[10:] + x[10:].square()))
    _, qa = mod.jacobian_qa(Toy())
    assert qa['fd_1e-05_relative_fro'] < 1e-10
    assert qa['fd_1e-06_relative_fro'] < 1e-10


def test_c_checkpoint_has_no_required_structured_numeric_export():
    paths = mod._source_paths(mod.DEFAULT_FULL, mod.DEFAULT_VERIFIED,
                             mod.DEFAULT_COUPLED, mod.DEFAULT_M3, mod.CHECKPOINT)
    assert not any(p.name == 'structured_numeric_export.json' for p in paths.values())
