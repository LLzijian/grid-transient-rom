import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from evaluate_local_memory_0098 import (TARGET_IDS, cached_targets, common_angle_embedding,
                                       jacobian_qa, modal, physical_output, OMEGA_S, energy_power_qa)


class SyntheticROM:
    def __init__(self, rank, offset=0.):
        self.dim = 30 + rank
        self.mass = torch.ones(10, dtype=torch.float64)
        self.offset = offset

    def field(self, x):
        delta, omega = x[..., :10], x[..., 10:20]
        acceleration = -(delta - delta.mean(-1, keepdim=True)) - .2 * omega
        return torch.cat((omega, acceleration, -x[..., 20:]), -1) + self.offset


@pytest.mark.parametrize("rank", [3, 10, 11])
def test_only_common_angle_removed_and_full_frequency_retained(rank):
    model = SyntheticROM(rank)
    matrix, qa = jacobian_qa(model)
    Q = common_angle_embedding(model.dim)
    assert qa["pass"]
    assert Q.shape == (30 + rank, 29 + rank)
    assert np.allclose(Q.T @ Q, np.eye(29 + rank))
    common_angle = np.r_[np.ones(10), np.zeros(20 + rank)]
    assert np.max(np.abs(Q.T @ common_angle)) < 1e-14
    common_speed = np.r_[np.zeros(10), np.ones(10), np.zeros(10 + rank)]
    assert np.allclose(Q @ Q.T @ common_speed, common_speed)
    assert np.linalg.norm(physical_output(Q.T @ common_speed, np.ones(10), Q)) > 0
    assert matrix.shape == (model.dim, model.dim)


def synthetic_cache(tmp_path):
    indices = np.arange(12, -1, -1)
    values = -.1 + 1j * np.arange(1, 14)
    vectors = np.zeros((20, 13), dtype=complex)
    vectors[10] = 1.
    outputs = np.column_stack([physical_output(vectors[:, i], np.ones(10), omega_scale=OMEGA_S) for i in indices])
    targets = [{"index": old, "real": float(values[i].real), "imag": float(values[i].imag),
                "frequency_hz": float(values[i].imag / (2 * np.pi)),
                "damping_ratio": float(-values[i].real / abs(values[i]))}
               for old, i in zip(TARGET_IDS, indices)]
    evidence = {"targets": targets, "integrity_pass": True, "hashes_unchanged": True,
                "thresholds": {"frequency_relative": .15, "damping_absolute": .05, "mac": .8}}
    (tmp_path / "evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
    np.savez(tmp_path / "modal_arrays.npz", target_recomputed_indices=indices,
             descriptor_eigenvalues=values, descriptor_right_vectors=vectors,
             target_ids=TARGET_IDS, target_outputs=outputs, mass=np.ones(10))
    return tmp_path


def test_cached_indices_not_original_target_ids_drive_alignment(tmp_path):
    cache = synthetic_cache(tmp_path)
    targets, outputs = cached_targets(cache, np.ones(10))
    assert targets[0]["imag"] == 13.
    assert len(outputs) == 13
    result = modal(SyntheticROM(11), cache=cache, output=tmp_path / "result")
    assert result["full_dimension"] == 41
    assert result["reduced_dimension"] == 40
    assert result["fom_recomputed"] is False
    assert result["source_hashes_before"] == result["source_hashes_after"]
    assert result["spectral_alpha"] == pytest.approx(-.1, abs=1e-12)
    assert "all_pair_metrics" in result


def test_modal_qa_failure_is_unresolved_not_zero_match_claim(tmp_path):
    result = modal(SyntheticROM(3, offset=.01), cache=synthetic_cache(tmp_path))
    assert not result["qa"]["pass"]
    assert result["matched_count"] is None
    assert result["spectral_alpha"] is None


def test_reject_stale_pole_mapping(tmp_path):
    cache = synthetic_cache(tmp_path)
    evidence = json.loads((cache / "evidence.json").read_text())
    evidence["targets"][0]["imag"] += .1
    (cache / "evidence.json").write_text(json.dumps(evidence))
    with pytest.raises(AssertionError, match="do not align"):
        cached_targets(cache, np.ones(10))


class SyntheticPowerROM(SyntheticROM):
    def interface_to_state(self, x):
        return x

    def energy(self, x):
        centered = torch.cat((x[..., :10] - x[..., :10].mean(-1, keepdim=True), x[..., 10:]), -1)
        return .5 * centered.square().sum(-1)

    def energy_and_gradient(self, x):
        e = torch.cat((x[..., :10] - x[..., :10].mean(-1, keepdim=True), x[..., 10:]), -1)
        return self.energy(x), e

    def matrices(self):
        J = torch.zeros((self.dim, self.dim), dtype=torch.float64)
        J[:10, 10:20] = torch.eye(10)
        J[10:20, :10] = -torch.eye(10)
        R = torch.diag(torch.cat((torch.zeros(10, dtype=torch.float64), torch.full((10,), .2, dtype=torch.float64), torch.ones(self.dim - 20, dtype=torch.float64))))
        return J, R

    def port_matrix(self):
        B = torch.zeros((self.dim, 10), dtype=torch.float64)
        B[10:20] = torch.eye(10)
        return B

    def port_values(self, x):
        return .05 * x[..., 10:20]

    def internal_field(self, x):
        return self.field(x) + self.port_values(x) @ self.port_matrix().T


def test_energy_power_includes_nonzero_actual_port_supply():
    result = energy_power_qa(SyntheticPowerROM(10))
    assert result["pass"]
    assert min(result["probes"]["actual_port_supply"]) > 0
    assert result["power_scaled_max_abs_error"] < 1e-12
