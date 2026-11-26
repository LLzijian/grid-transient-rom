"""Synthetic-only checks of edge-agreement statistics."""

import json
from math import comb

import numpy as np
import pytest

from experiments.osgkd_v2_wp0_20261127 import edge_audit as audit


def test_average_ranks_holm_and_exact_overlap_null():
    values = np.arange(45, dtype=float)
    values[:3] = 0
    np.testing.assert_array_equal(audit.average_ranks(values)[:4], [2, 2, 2, 4])
    assert audit.holm([.04, .01, .03]) == [.06, .03, .06]
    null = audit.overlap_null(12, 12)
    assert null['p_one_sided'] == pytest.approx(1/comb(45, 12))
    assert null['expected_I'] == pytest.approx(144/45)
    assert null['expected_F1'] == pytest.approx(12/45)
    # Independent enumeration of the hypergeometric distribution.
    weights = np.array([comb(12, i)*comb(33, 12-i) for i in range(13)], dtype=float)
    weights /= weights.sum()
    assert null['expected_J'] == pytest.approx(sum(weights[i]*i/(24-i) for i in range(13)))
    with pytest.raises(ValueError):
        audit.average_ranks([float('nan')]*45)


def test_identical_asymmetric_graphs_and_constant_ties():
    rng = np.random.default_rng(501)
    values = rng.permutation(45).astype(float)
    result = audit.statistics({key: values for key in audit.MODEL_KEYS})
    assert result['strong_rank']
    assert all(result['stable_K'].values())
    assert result['holm_family_size'] == 12
    assert result['permutation_seed'] == 27031
    assert result['permutation_draws'] == 9999
    for pair in result['pairs']:
        assert pair['rank']['rho'] == pytest.approx(1)
        assert pair['rank']['holm_p'] < .05
        for k, top in pair['top'].items():
            assert top['I'] == int(k)
            assert top['J'] == top['F1'] == 1
            assert top['holm_p'] < .05
            assert not top['boundary_tie_first']
    tied = audit.statistics({key: np.ones(45) for key in audit.MODEL_KEYS})
    assert not tied['strong_rank']
    assert not any(tied['stable_K'].values())
    for pair in tied['pairs']:
        assert pair['rank']['rho'] is None
        assert pair['rank']['holm_p'] == 1
        assert all(top['permutation_p'] == 1 and top['boundary_tie_first']
                   for top in pair['top'].values())
    json.dumps([result, tied], allow_nan=False)
    np.testing.assert_array_equal(audit.top_indices(np.ones(45), 12), np.arange(12))
