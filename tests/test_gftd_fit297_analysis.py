"""Engineering fixtures only: no fit297, actual model parameters, or 699D pencil."""
from fractions import Fraction as F
from math import factorial
import importlib.util
from pathlib import Path
import zipfile

import numpy as np
import pytest

from kiki.physics.kuramoto_functional import GeneratingFunctionalKuramoto, StructuralGates

spec = importlib.util.spec_from_file_location("fit297_analysis", Path(__file__).parents[1]/"experiments/gftd_fit297_analysis_20261107.py")
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def functional(*, modulation=.1, eta=0.):
    n = 2
    return GeneratingFunctionalKuramoto(
        mass=np.ones(n), damping=np.ones(n), k_edge=np.ones(1), eta_edge=np.array([eta]),
        gamma_amp=np.ones(n)*.1, lambda_amp=np.ones(n), lambda_mem=np.ones(n),
        tau_amp=np.ones(n), tau_mem_sync=np.ones(n), tau_mem_coi=1.,
        j_omega_mem=np.ones(n), gates=StructuralGates(np.ones(1), np.array([modulation]), np.ones(n)))


def test_exact_stored_factors_and_rounded_center():
    f = functional(eta=np.nextafter(.1, 1))
    f.k_edge[:] = .1
    f.gates.s_edge[:] = .3
    cert = a.l2_certificate(f, [.1, 0.])
    assert cert["status"] == "certified"
    assert F(cert["exact_K"][0]) == F.from_float(.1)*F.from_float(.3)
    assert F(cert["exact_K"][0]) != F.from_float(float(.1*.3))
    assert len(cert["candidates"]) == 13
    assert F(cert["selected"]["g"]) > 0
    z = np.r_[[.1, 0.], np.zeros(8)]
    point = a.l2_point(cert, f, z)
    assert point["status"] == "certified"
    assert point["epsilon"] is None
    assert point["true_initialization_set_coverage"] == "unknown"
    f.k_edge[:] = .2
    assert a.l2_point(cert, f, z)["reason"] == "functional_changed_after_certificate"


def test_l2_connectivity_dissipation_and_strict_ball():
    f = functional()
    f.k_edge[:] = 0
    assert a.l2_certificate(f, [0., 0.])["reason"] == "active_graph_disconnected"
    f.k_edge[:] = 1
    f.damping[:] = 0
    assert a.l2_certificate(f, [0., 0.])["status"] == "not_certified"
    f.damping[:] = 1
    cert = a.l2_certificate(f, [0., 0.])
    R = F(cert["selected"]["R"])
    z = [F(0)]*10
    z[2] = R
    assert a.l2_point(cert, f, z)["reason"] == "point_outside_strict_ball"
    assert a.exact_quotient_norm_squared([F(3), F(3)]+[F(0)]*8, [0, 0]) == 0


def test_taylor_bounds_fixed_remainders_and_signed_intervals():
    import decimal
    decimal.getcontext().prec = 70
    for x in (F(-1, 3), F(0), F(1, 3)):
        lo, hi = a.taylor_interval(x, "exp")
        exact = decimal.Decimal(x.numerator)/decimal.Decimal(x.denominator)
        value = exact.exp()
        assert decimal.Decimal(lo.numerator)/decimal.Decimal(lo.denominator) <= value
        assert decimal.Decimal(hi.numerator)/decimal.Decimal(hi.denominator) >= value
        for kind, degree in (("sin", 38), ("cos", 37)):
            lo, hi = a.taylor_interval(x, kind)
            assert hi-lo == 2*abs(x)**degree/factorial(degree)
    assert a._mul((F(-2), F(3)), (F(-4), F(5))) == (F(-12), F(15))
    with pytest.raises(ValueError):
        a.taylor_interval(F(1), "exp")


def case_tables():
    line = dict(bus1=1, bus2=2, Vn1=10., Vn2=10., Sn=100., u=1., tap=2., phi=0., r=0., x=1.,
                g=0., b=0., g1=0., b1=0., g2=0., b2=0.)
    return dict(Bus=[dict(idx=i, Vn=10., v0=1.) for i in (1, 2, 3)],
                Line=[line, dict(line, bus1=2, bus2=3, tap=1.)],
                Shunt=[dict(bus=1, Vn=10., Sn=100., u=1., g=1., b=0.)],
                PQ=[dict(bus=3, u=1., p0=2., q0=-.5)])


def test_bus1_tap_negative_q_and_schur():
    Y, ids = a.assemble_ybus(case_tables())
    assert Y[0, 0] == 1-.25j
    assert Y[0, 1] == .5j
    assert Y[1, 0] == .5j
    assert Y[2, 2] == 2-.5j
    result = a.schur_reference(Y, ids, [2, 3])
    assert result["status"] == "qualified"
    assert result["solve_count"] == result["svd_count"] == 1
    red = np.array(result["Yred_real"])+1j*np.array(result["Yred_imag"])
    assert np.allclose(red, Y[1:, 1:]-np.outer(Y[1:, 0], Y[0, 1:])/Y[0, 0])
    assert a.schur_reference(np.zeros((2, 2)), [1, 2], [2])["reason"] == "singular_eliminated_block"


def test_workbook_rejects_other_bytes(tmp_path):
    path = tmp_path/"wrong.xlsx"
    path.write_bytes(b"fixture")
    with pytest.raises(ValueError, match="case_sha256"):
        a.read_case_workbook(path)


def test_zip_xml_reader_small_fixture(tmp_path, monkeypatch):
    path = tmp_path/"small.xlsx"
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/workbook.xml", f'<workbook xmlns="{ns}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Bus" r:id="rId1"/></sheets></workbook>')
        archive.writestr("xl/_rels/workbook.xml.rels", '<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>')
        archive.writestr("xl/worksheets/sheet1.xml", f'<worksheet xmlns="{ns}"><sheetData><row><c r="A1" t="inlineStr"><is><t>idx</t></is></c><c r="B1" t="inlineStr"><is><t>Vn</t></is></c></row><row><c r="A2"><v>1</v></c><c r="B2"><v>10</v></c></row></sheetData></worksheet>')
    monkeypatch.setattr(a, "CASE_SHA", a._sha(path))
    assert a.read_case_workbook(path) == {"Bus": [{"idx": 1., "Vn": 10.}]}


def test_small_descriptor_two_qz_and_complete_matching():
    A = np.array([[-.2, -2.], [2., -.2]])
    result = a.descriptor_modes(A, np.eye(2), [0, 1], [0, 1], np.ones(2), 1., np.array([-.2+2j]))
    assert result["status"] == "qualified"
    assert result["checks"]["qz_count"] == 2
    assert len(result["modes"]) == 1
    reference = dict(result, modes=[dict(result["modes"][0], frequency_hz=.02+i*.01) for i in range(13)])
    candidate = dict(result, modes=[dict(result["modes"][0], frequency_hz=5.+i) for i in range(14)])
    matched = a.match_modes(reference, candidate)
    assert len(matched["matches"]) == 13
    assert all(m["relative_frequency_error"] > .5 for m in matched["matches"])
    assert len(matched["unmatched_candidate"]) == 1
    assert matched["unmatched_reference"] == []
    failed = a.descriptor_modes(np.zeros((2, 2)), np.zeros((2, 2)), [0], [0], np.ones(1), 1., [])
    assert failed["status"] == "unknown"


def test_small_rom_only_removes_common_theta():
    f = functional()
    result = a.rom_modes(f, [0., 0.])
    assert result["status"] == "qualified", result
    assert result["dimension"] == 9
    assert result["retained_omega"] == 2
    assert result["counters"]["eig_count"] == 2
    assert result["counters"]["rhs_calls"] == 38
    cert = a.l2_certificate(f, [0., 0.])
    assert a.rom_modes(f, [0., 0.], cert)["status"] == "qualified"


def test_FOM_identity_failure_does_not_compute_spectrum(tmp_path, monkeypatch):
    path = tmp_path/"data"
    path.write_bytes(b"fixture")
    monkeypatch.setattr(a, "descriptor_modes", lambda *args: pytest.fail("unexpected QZ"))
    result = a.fom_modes(path, path, path, np.ones(10), expected_identity={})
    assert result["reason"] == "FOM_source_hash_mismatch"


def test_gated_dissipation_exact_products_and_domain_failure():
    f = functional()
    f.gate_weight_amp = np.zeros((2, 3))
    f.j_omega_mem[:] = .1
    f.lambda_mem[:] = .3
    cert = a.l2_certificate(f, [0., 0.])
    assert cert["status"] == "certified"
    assert F(cert["dissipation_lower"]["amp"][0]) == F(1, 1+3**40)
    assert F(cert["exact_operator_products"]["J_Kr"][0]) == F.from_float(.1)*F.from_float(.3)
    f.eta_edge[:] = 2.
    assert a.l2_certificate(f, [0., 0.])["reason"] == "no_positive_radius_cap"


def test_graph_weights_and_reference_stiffness():
    f = functional(modulation=.25)
    kron = dict(status="qualified", Yred_real=[[1., -2.], [-2., 1.]], Yred_imag=[[0., 0.], [0., 0.]])
    result = a.graph_correspondence(kron, f, [.1, 0.])
    assert result["spearman"] is None
    assert result["effective_weights"] == [1.]
    edge = result["edges"][0]
    assert F(edge["exact_beta"]) == F(1, 4)
    assert edge["reference_angle_stiffness"] == pytest.approx(np.cos(.1))


def test_FOM_device_ID_order_matches_static_audit(tmp_path, monkeypatch):
    import json
    descriptor, evidence, accepted = [tmp_path/name for name in ("descriptor.npz", "evidence.json", "accepted.json")]
    descriptor.write_bytes(b"not-read")
    identity = dict(andes="2.0.0", case_sha256=a.CASE_SHA, physical_signature=a.SIGNATURE)
    evidence.write_text(json.dumps(identity), encoding="utf-8")
    accepted.write_text("{}", encoding="utf-8")
    hashes = {descriptor: a.DESCRIPTOR_SHA, evidence: a.EVIDENCE_SHA, accepted: a.ACCEPTED_SHA}
    monkeypatch.setattr(a, "_sha", lambda path: hashes[Path(path)])
    monkeypatch.setattr(a, "descriptor_modes", lambda *args: pytest.fail("unexpected QZ"))
    expected = dict(identity, generator_order=[f"GENROU_{i}" for i in range(1, 11)])
    result = a.fom_modes(descriptor, evidence, accepted, np.ones(10), expected_identity=expected)
    assert result["reason"] == "fit297_equilibrium_identity_unverified"
    expected["generator_order"] = list(range(1, 11))
    result = a.fom_modes(descriptor, evidence, accepted, np.ones(10), expected_identity=expected)
    assert result["reason"] == "fit297_generator_order_unverified"
