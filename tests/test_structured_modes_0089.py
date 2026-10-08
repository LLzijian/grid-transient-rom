import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from scipy.linalg import null_space

from experiments.compare_physical_modes import assignment, ordinary_modes, physical_output
from experiments.audit_structured_modes_0089 import validate_cached_objects, validate_inherited, BASE_MODEL_SHA, CANDIDATE_MODEL_SHA


def test_common_angle_quotient_retains_frequency_block():
    q = np.zeros((33, 32)); q[:10, :9] = null_space(np.ones((1, 10))); q[10:, 9:] = np.eye(23)
    assert np.max(np.abs(q.T @ q - np.eye(32))) < 1e-12
    assert np.array_equal(q[10:, 9:], np.eye(23))
    assert np.max(np.abs(q[:10, 9:])) == 0


def test_two_cached_style_eigensystems_have_residuals():
    rng = np.random.default_rng(89); a = rng.normal(size=(32, 32)) - 3 * np.eye(32)
    values, vectors, residual = ordinary_modes(a)
    assert residual <= 1e-8
    assert np.isfinite(values).all() and np.isfinite(vectors).all()


def test_physical_output_preserves_common_frequency():
    q = np.zeros((33, 32)); q[:10, :9] = null_space(np.ones((1, 10))); q[10:, 9:] = np.eye(23)
    v = np.zeros(32); v[9:19] = 1.0
    out = physical_output(v, np.ones(10), q)
    assert np.allclose(out[10:20], np.ones(10) / 1.0033667249602924)


def test_cached_validator_rejects_wrong_object_and_removed_common_frequency():
    q = np.zeros((33, 32)); q[:10, :9] = null_space(np.ones((1, 10))); q[10:, 9:] = np.eye(23)
    a = -np.eye(33); a[:, :10] = 0; a32 = q.T @ a @ q
    validate_cached_objects(a, q, a32, np.ones(10))
    bad = a.copy(); bad[20, 20] = 1e-2
    try: validate_cached_objects(bad, q, a32, np.ones(10))
    except AssertionError: pass
    else: raise AssertionError("wrong cached object accepted")
    qbad = q.copy(); qbad[10:, 9] = 0
    try: validate_cached_objects(a, qbad, a32, np.ones(10))
    except AssertionError: pass
    else: raise AssertionError("removed common-frequency block accepted")


def test_inherited_validator_rejects_missing_flags_and_wrong_sha():
    flags = {k: True for k in ("finite", "origin", "gauge", "common_angle", "Q", "Q_common_angle", "eigen_residual", "fd_ad", "fd_spectral", "sensitivity")}
    e83 = {"models": {"baseline": {"qa": {'flags': flags}}}}
    e88 = {"candidate_hashes": {"model.pt": CANDIDATE_MODEL_SHA}, "candidate_qa": {
        'actual_relative': 0., "affinity": {"candidate_affinity_relative": 0.0, 'direct_column_relative_max':0.},
        "structural": {k:0. for k in ('origin_max', 'kinematic_max', 'gauge_field_relative', 'full_reload_relative', 'numeric_export_relative', 'power_identity_relative', 'J_antisymmetry', 'R_symmetry', 'R_min_eigenvalue')}}}
    before = {"base_model": BASE_MODEL_SHA, "candidate_model": CANDIDATE_MODEL_SHA}
    validate_inherited({"pair_difference_count": 0, "max_pair_numeric_difference": 0.0, "matched_count": 1, "reported_matched_count": 1}, e83, e88, before)
    bad = dict(e83); bad["models"] = {"baseline": {"qa": {}}}
    try: validate_inherited({"pair_difference_count": 0, "max_pair_numeric_difference": 0.0, "matched_count": 1, "reported_matched_count": 1}, bad, e88, before)
    except AssertionError: pass
    else: raise AssertionError("missing inherited QA accepted")
    bad_before = dict(before, candidate_model="wrong")
    try: validate_inherited({"pair_difference_count": 0, "max_pair_numeric_difference": 0.0, "matched_count": 1, "reported_matched_count": 1}, e83, e88, bad_before)
    except AssertionError: pass
    else: raise AssertionError("wrong SHA accepted")


def test_missing_or_nan_prior_error_is_not_passing():
    from experiments.audit_structured_modes_0089 import bounded
    assert not bounded(None, 1e-10)
    assert not bounded(float('nan'), 1e-10)
    assert not bounded(-1., 1e-10)
    assert bounded(0., 1e-10)


def test_source_link_rejects_changed_cache_source():
    from experiments.audit_structured_modes_0089 import validate_links
    import pytest
    old = {'hashes_before': {'descriptor':'unchanged'}, 'hashes_after': {'descriptor':'unchanged'}}
    with pytest.raises(AssertionError, match='0078 source link'):
        validate_links(old, old, old, {'old78_descriptor':'changed'})
