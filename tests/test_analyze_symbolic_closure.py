import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))

from analyze_symbolic_closure import analyze_model, quotient_basis, run_cli
from structured_rom import StructuredROM
from symbolic_closure_rom import SymbolicClosureROM


class SmoothQuadraticToy:
    """Synthetic 40D model with exact quadratic energy and linear field."""

    dim = 40

    def __init__(self, hessian, jacobian, field_offset=None, energy_offset=0.0):
        self.hessian = torch.as_tensor(hessian, dtype=torch.float64)
        self.jacobian = torch.as_tensor(jacobian, dtype=torch.float64)
        self.field_offset = torch.as_tensor(
            np.zeros(40) if field_offset is None else field_offset, dtype=torch.float64
        )
        self.energy_offset = float(energy_offset)

    def parameters(self):
        return iter(())

    def energy(self, x):
        return 0.5 * x @ self.hessian @ x + self.energy_offset

    def internal_field(self, x):
        return self.jacobian @ x + self.field_offset


def _stable_matrices(indefinite=False, unstable=False):
    Q, _ = quotient_basis()
    values = np.linspace(0.7, 2.4, 39)
    if indefinite:
        values[0] = -0.4
    H = Q @ np.diag(values) @ Q.T
    rates = -np.linspace(0.25, 1.3, 39)
    if unstable:
        rates[-1] = 0.18
    A = Q @ np.diag(rates) @ Q.T
    return H, A


def test_fixed_quotient_basis_is_orthonormal_and_removes_only_common_angle():
    Q, gauge = quotient_basis()
    np.testing.assert_allclose(Q.T @ Q, np.eye(39), atol=2e-15, rtol=0)
    np.testing.assert_allclose(Q @ Q.T, np.eye(40) - np.outer(gauge, gauge), atol=2e-15, rtol=0)
    np.testing.assert_allclose(Q.T @ gauge, np.zeros(39), atol=2e-15, rtol=0)


def test_positive_energy_curvature_and_strict_hurwitz_have_local_witness():
    H, A = _stable_matrices()
    result, arrays = analyze_model(SmoothQuadraticToy(H, A))
    assert result["finite"] == {"H": True, "A": True, "K_projected": True, "A_projected": True}
    assert result["energy_curvature"]["raw_strict_positive_definite"] is False
    assert result["energy_curvature"]["quotient_strict_positive_definite"] is True
    assert result["linear_field"]["strict_hurwitz"] is True
    assert result["linear_field"]["strictly_stable_mode_count"] == 39
    assert result["origin"]["equilibrium_pass"] is True
    assert result["origin"]["energy_stationary_pass"] is True
    assert result["origin"]["quotient_eligible"] is True
    assert result["origin"]["H0"] == 0.0
    assert result["origin"]["gradH0_norm_inf"] == 0.0
    assert result["origin"]["f0_norm_inf"] == 0.0
    assert result["lyapunov"]["verified"] is True
    assert result["lyapunov"]["interpretation"] == (
        "linearized quadratic witness only; nonlinear radius not certified"
    )
    np.testing.assert_allclose(arrays["H"] @ arrays["gauge"], 0, atol=2e-14, rtol=0)
    np.testing.assert_allclose(arrays["A"] @ arrays["gauge"], 0, atol=2e-14, rtol=0)


def test_indefinite_energy_and_positive_real_part_are_reported_separately():
    H, A = _stable_matrices(indefinite=True, unstable=True)
    result, _ = analyze_model(SmoothQuadraticToy(H, A))
    assert result["energy_curvature"]["quotient_strict_positive_definite"] is False
    assert result["energy_curvature"]["quotient_positive_eigenvalue_count"] == 38
    assert result["linear_field"]["strict_hurwitz"] is False
    assert result["linear_field"]["strictly_unstable_mode_count"] == 1
    assert result["lyapunov"]["attempted"] is False
    assert result["scope"].startswith("local derivatives")


def test_nonfinite_hessian_is_retained_without_attempting_its_eigendecomposition():
    H, A = _stable_matrices()
    H[0, 0] = np.nan
    result, arrays = analyze_model(SmoothQuadraticToy(H, A))
    assert result["finite"]["H"] is False
    assert result["raw_H_eigenvalues_real_imag"] is None
    assert result["energy_curvature"]["quotient_strict_positive_definite"] is False
    assert result["lyapunov"]["attempted"] is False
    assert np.isnan(arrays["H"]).any()


def test_full_minus_identity_breaks_common_angle_gauge_and_skips_lyapunov():
    Q, _ = quotient_basis()
    H = Q @ np.diag(np.ones(39)) @ Q.T
    A = -np.eye(40)
    result, arrays = analyze_model(SmoothQuadraticToy(H, A))
    assert result["linear_field"]["strict_hurwitz"] is True
    assert result["origin"]["equilibrium_pass"] is True
    assert result["quotient"]["gauge_invariance_pass"] is False
    assert result["origin"]["quotient_eligible"] is False
    assert result["lyapunov"]["attempted"] is False
    assert "algebraic spectrum only" in result["linear_field"]["interpretation"]
    np.testing.assert_allclose(arrays["A"], -np.eye(40), atol=0, rtol=0)


def test_affine_nonzero_field_is_not_a_fixed_point_witness_even_if_jacobian_is_hurwitz():
    H, A = _stable_matrices()
    offset = np.zeros(40)
    offset[20] = 0.02
    result, arrays = analyze_model(SmoothQuadraticToy(H, A, field_offset=offset, energy_offset=1.25))
    assert result["linear_field"]["strict_hurwitz"] is True
    assert result["origin"]["H0"] == 1.25
    assert result["origin"]["f0_norm_inf"] == 0.02
    assert result["origin"]["equilibrium_pass"] is False
    assert result["origin"]["quotient_eligible"] is False
    assert result["lyapunov"]["attempted"] is False
    assert result["lyapunov"]["reason_skipped"]
    np.testing.assert_allclose(arrays["f0"], offset, atol=0, rtol=0)


def _synthetic_export(path):
    eye = torch.eye(10, dtype=torch.float64)
    D = torch.zeros((10, 10), dtype=torch.float64)
    D[:] = eye - torch.roll(eye, shifts=-1, dims=0)
    base = StructuredROM(
        mass=torch.linspace(1.0, 1.9, 10), D=D,
        coupling=torch.linspace(0.2, 0.6, 10),
        delta_star=torch.linspace(-0.08, 0.09, 10),
        damping=torch.full((10,), 0.12), gain=torch.ones(10),
        jpi_p=torch.full((10,), 0.3), rpp=torch.full((10,), 0.08),
        n_q=0, storage=(),
    ).double()
    model = SymbolicClosureROM(base, input_scale=torch.ones(20), p_scale=torch.ones(10), seed=7)
    model.export(path)


def test_cli_reads_only_standard_model_export_and_writes_summary_and_matrices(tmp_path):
    model_path = tmp_path / "synthetic_model.json"
    _synthetic_export(model_path)
    out = tmp_path / "analysis"
    result = run_cli([model_path], out)
    assert result["model_count"] == 1
    record = result["models"][0]
    assert len(record["source_sha256"]) == 64
    assert record["source_path"] == str(model_path.resolve())
    assert record["coordinates"].startswith("internal")
    assert (out / "summary.json").is_file()
    assert (out / record["matrix_file"]).is_file()
    saved = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert saved["data_access"].startswith("none")
    with np.load(out / record["matrix_file"]) as matrices:
        assert matrices["H"].shape == (40, 40)
        assert matrices["A"].shape == (40, 40)
        assert matrices["Q"].shape == (40, 39)
