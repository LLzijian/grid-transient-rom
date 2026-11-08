"""Run3 model-only analysis. Importing this module performs no scientific work."""
from __future__ import annotations

from fractions import Fraction as F
from math import factorial
from pathlib import Path
import hashlib
import json
import re
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
from scipy.linalg import eig, block_diag
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from kiki.evaluation.modal import DmdMode, match_dmd_modes, modal_assurance_criterion

CASE_SHA = "9c2048dc94201ee48ffe816de65d53831fa1f4b0e7fdf32f24c50f1017edcfe5"
DESCRIPTOR_SHA = "6648fca0340fbd0823b5f0d50e99f19ffa00ce6ee32bfb36002219938371c4da"
EVIDENCE_SHA = "37d82fb8d1c3af8bb5d4cd7266c14d04913b10aa7caec77ac8beeb29336435fa"
ACCEPTED_SHA = "8a1d3e53a1b4aeba5ceb2f1c7017378dc3ceb51be64a372a3562da8b9a7970ef"
SIGNATURE = "e52c4d30ec9db0ae76c6a2f1f722ca00a7a5803c4dbb01ad3b6718bfcade16d9"


def fraction(value):
    if isinstance(value, (F, str, int)):
        return F(value)
    return F.from_float(float(value))


def _qarray(values):
    return [fraction(v) for v in values]


def _fail(reason, **details):
    return dict(status="unknown", reason=reason, **details)


def _l2_fail(reason, **details):
    return dict(status="not_certified", reason=reason,
                true_initialization_set_coverage="unknown", **details)


def _functional_hash(f):
    from dataclasses import fields
    payload = {}
    for field in fields(f):
        value = getattr(f, field.name)
        if field.name == "gates":
            value = {k: np.asarray(getattr(value, k)).tolist() for k in ("s_edge", "s_mod", "s_mem", "s_coi")}
        elif value is not None:
            value = np.asarray(value).tolist()
        payload[field.name] = value
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _parameters(f, delta_star):
    n = f.n_nodes
    theta = _qarray(delta_star)
    if len(theta) != n or n < 2:
        raise ValueError("delta_star dimension or node count")
    mean = sum(theta) / n
    theta = [t - mean for t in theta]
    K = [fraction(k) * fraction(s) for k, s in zip(f.k_edge, f.gates.s_edge, strict=True)]
    beta = [k * fraction(s) for k, s in zip(K, f.gates.s_mod, strict=True)]
    gamma = _qarray(f.gamma_amp)
    eta = _qarray(f.eta_edge)
    phi = [theta[i]-theta[j] for i, j in f.edges]
    return dict(n=n, theta=theta, K=K, beta=beta, gamma=gamma, eta=eta,
                delta=[abs(a-b) for a, b in zip(phi, eta, strict=True)],
                phi=phi, Gamma=[abs(gamma[i])+abs(gamma[j]) for i, j in f.edges])


def l2_certificate(functional, delta_star):
    """Exactly 13 rational candidates; no data-dependent radius selection."""
    f = functional
    try:
        p = _parameters(f, delta_star)
        n, K, beta = p["n"], p["K"], p["beta"]
        if any(not 0 <= b <= k for k, b in zip(K, beta, strict=True)):
            return _l2_fail("edge_parameter_domain")
        positive = ("mass", "storage_p", "lambda_amp", "lambda_mem", "damping", "rate_p", "tau_amp", "tau_mem_sync")
        arrays = {name: _qarray(getattr(f, name)) for name in positive}
        if any(len(v) != n or min(v) <= 0 for v in arrays.values()):
            return _l2_fail("strict_storage_or_dissipation_missing")
        active = [e for e, k in enumerate(K) if k > 0]
        connected = {0}
        for _ in range(n):
            for e in active:
                i, j = f.edges[e]
                if i in connected or j in connected:
                    connected.update((i, j))
        if len(connected) != n:
            return _l2_fail("active_graph_disconnected")
        # Clipped sigmoid >= 1/(1+exp(40)) > 1/(1+3**40), since e<3.
        gate_lower = F(1, 1 + 3**40)
        dissipation = {}
        for port, tau in (("amp", "tau_amp"), ("mem", "tau_mem_sync")):
            lower = gate_lower if getattr(f, "gate_weight_"+port) is not None else F(1)
            dissipation[port] = [str(lower / t) for t in arrays[tau]]
        A0 = max(max(abs(p["phi"][e]), abs(p["eta"][e])) for e in active)
        Gmax = max(p["Gamma"][e] for e in active)
        cap = min(F(1), (F(7, 5)-A0)/3)
        if Gmax:
            cap = min(cap, 1/(2*Gmax))
        if cap <= 0:
            return _l2_fail("no_positive_radius_cap")
        ell = 4*min(K[e] for e in active)/(n*(n-1))
        g = F(3, 2)*sum(K[e]*p["delta"][e] for e in active)
        g += F(1, 2)*sum(beta[e]*p["Gamma"][e]*p["delta"][e]**2 for e in active)
        ma = min(arrays["lambda_amp"])
        candidates = []
        for j in range(13):
            R = cap / 2**j
            alpha, h = A0+F(3, 2)*R, R*Gmax
            row = dict(index=j, R=str(R), status="not_certified")
            if not (alpha < F(7, 5) and h < 1):
                row["reason"] = "angle_or_exponential_domain"
            else:
                cos_lower = 1-alpha**2/2
                mt = cos_lower*(1-h)*ell
                C = sum(beta[e]*p["Gamma"][e]*(2*R+F(3, 2)*p["delta"][e]) for e in active)/(1-h)
                det = mt*ma-C*C
                if cos_lower <= 0 or det <= 0:
                    row["reason"] = "curvature_lower_bound_nonpositive"
                else:
                    mu = min(det/(mt+ma), min(arrays["mass"]), min(arrays["storage_p"]), min(arrays["lambda_mem"]))
                    if not g < mu*R/4:
                        row["reason"] = "center_gradient_gate"
                    else:
                        rho = mu*R*R/4
                        # Row sums bound the positive 2x2 Hessian comparison matrix.
                        A = 2*sum(K)/(1-h)
                        D = max(arrays["lambda_amp"])+sum(beta[e]*p["Gamma"][e]**2*(p["delta"][e]+F(3, 2)*R)**2 for e in active)/(2*(1-h))
                        L = max(A+C, D+C, max(arrays["mass"]), max(arrays["storage_p"]), max(arrays["lambda_mem"]))
                        row.update(status="certified", mu=str(mu), rho=str(rho), g=str(g),
                                   h=str(h), mixed_upper=str(C), hessian_upper=str(L))
            candidates.append(row)
        eligible = [r for r in candidates if r["status"] == "certified"]
        if not eligible:
            return _l2_fail("no_candidate_passed", candidates=candidates)
        selected = max(eligible, key=lambda r: (F(r["rho"]), -F(r["R"])))
        return dict(status="certified", selected=selected, candidates=candidates,
                    functional_sha256=_functional_hash(f),
                    center_theta=[str(t) for t in p["theta"]],
                    exact_K=[str(k) for k in K], exact_beta=[str(b) for b in beta],
                    dissipation_lower=dissipation,
                    exact_operator_products={
                        "J_Kr": [str(fraction(s)*fraction(j)*fraction(kr)) for s, j, kr in zip(f.gates.s_mem, f.j_omega_mem, f.lambda_mem, strict=True)],
                        "Kr_over_tau_mem": [str(fraction(kr)/fraction(t)) for kr, t in zip(f.lambda_mem, f.tau_mem_sync, strict=True)],
                        "gain_Sp": [str(fraction(gain)*fraction(sp)) for gain, sp in zip(f.gain_p, f.storage_p, strict=True)]},
                    actual_point_count=0,
                    dimension=5*n-1, scope="ROM attraction to equilibrium phase orbit",
                    true_initialization_set_coverage="unknown", epsilon=None)
    except (ValueError, TypeError, OverflowError, ZeroDivisionError) as exc:
        return _l2_fail("invalid_parameters", detail=str(exc))


def _add(a, b):
    return (a[0]+b[0], a[1]+b[1])


def _scale(a, s):
    return (a[0]*s, a[1]*s) if s >= 0 else (a[1]*s, a[0]*s)


def _mul(a, b):
    products = [x*y for x in a for y in b]
    return min(products), max(products)


def taylor_interval(x, kind):
    """Fixed protocol degrees and Lagrange/geometric remainders, all rational."""
    x = fraction(x)
    if kind == "exp":
        if abs(x) >= 1:
            raise ValueError("exp interval requires |x|<1")
        value = sum(x**k/factorial(k) for k in range(19))
        error = abs(x)**19/(factorial(19)*(1-abs(x)))
    elif kind in ("sin", "cos"):
        offset = int(kind == "sin")
        value = sum((-1)**k*x**(2*k+offset)/factorial(2*k+offset) for k in range(19))
        degree = 38 if offset else 37
        error = abs(x)**degree/factorial(degree)
    else:
        raise ValueError("unknown Taylor function")
    return value-error, value+error


def exact_quotient_norm_squared(state, center_theta):
    values, theta = _qarray(state), _qarray(center_theta)
    n = len(theta)
    if len(values) != 5*n:
        raise ValueError("state dimension")
    d = [values[i]-theta[i] for i in range(n)]
    return sum(x*x for x in d)-sum(d)**2/n+sum(x*x for x in values[n:])


def l2_point(certificate, functional, state):
    """One exact envelope for one qualified returned point; epsilon stays unknown."""
    if certificate.get("status") != "certified":
        return _l2_fail("model_not_certified")
    try:
        f = functional
        if certificate.get("functional_sha256") != _functional_hash(f):
            return _l2_fail("functional_changed_after_certificate")
        n = f.n_nodes
        z = _qarray(state)
        row = certificate["selected"]
        R, mu, rho, g = (F(row[k]) for k in ("R", "mu", "rho", "g"))
        norm2 = exact_quotient_norm_squared(z, certificate["center_theta"])
        if norm2 >= R*R:
            return _l2_fail("point_outside_strict_ball", norm_squared=str(norm2))
        K, beta = _qarray(certificate["exact_K"]), _qarray(certificate["exact_beta"])
        theta, omega, p, a, r = [z[i*n:(i+1)*n] for i in range(5)]
        energy = F(0)
        for block, name in ((omega, "mass"), (p, "storage_p"), (a, "lambda_amp"), (r, "lambda_mem")):
            energy += sum(fraction(c)*v*v/2 for c, v in zip(getattr(f, name), block, strict=True))
        H = (energy, energy)
        gamma, eta = _qarray(f.gamma_amp), _qarray(f.eta_edge)
        for e, (i, j) in enumerate(f.edges):
            if K[e] == 0:
                continue
            phi = theta[i]-theta[j]
            psi = _add(taylor_interval(eta[e], "cos"), _scale(taylor_interval(phi, "cos"), -1))
            psi = _add(psi, _scale(taylor_interval(eta[e], "sin"), -(phi-eta[e])))
            exp = taylor_interval(gamma[i]*a[i]+gamma[j]*a[j], "exp")
            coefficient = _add((K[e]-beta[e], K[e]-beta[e]), _scale(exp, beta[e]))
            H = _add(H, _mul(coefficient, psi))
        upper = H[1]+g*g/(2*mu)
        return dict(status="certified" if upper < rho else "not_certified",
                    reason=None if upper < rho else "energy_margin_unresolved",
                    energy_interval=[str(v) for v in H], V_upper=str(upper),
                    margin_lower=str(1-upper/rho), norm_squared=str(norm2),
                    true_initialization_set_coverage="unknown", epsilon=None)
    except (ValueError, TypeError, OverflowError, ZeroDivisionError) as exc:
        return _l2_fail("point_envelope_failed", detail=str(exc))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_case_workbook(path):
    """Read only the hash-bound official workbook using ZIP/XML, no Excel engine."""
    if _sha(path) != CASE_SHA:
        raise ValueError("case_sha256_mismatch")
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rel_ns = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
    result = {}
    with zipfile.ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            strings = ["".join(node.itertext()) for node in root.findall("m:si", ns)]
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {r.attrib["Id"]: r.attrib["Target"] for r in rels}
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        for sheet in workbook.findall("m:sheets/m:sheet", ns):
            name = sheet.attrib["name"]
            if name not in ("Bus", "Line", "Shunt", "PQ", "GENROU"):
                continue
            target = targets[sheet.attrib[rel_ns]]
            target = target.lstrip("/") if target.startswith("/") else "xl/"+target
            root = ET.fromstring(archive.read(target))
            rows = []
            for row in root.findall("m:sheetData/m:row", ns):
                cells = {}
                for cell in row.findall("m:c", ns):
                    letters = re.match(r"[A-Z]+", cell.attrib["r"])[0]
                    column = 0
                    for letter in letters:
                        column = column*26+ord(letter)-64
                    value = cell.find("m:v", ns)
                    kind = cell.attrib.get("t")
                    if kind == "inlineStr":
                        parsed = "".join(cell.find("m:is", ns).itertext())
                    elif value is None:
                        continue
                    elif kind == "s":
                        parsed = strings[int(value.text)]
                    elif kind in ("str", "e"):
                        parsed = value.text
                    else:
                        parsed = float(value.text)
                    cells[column-1] = parsed
                rows.append(cells)
            header = rows[0]
            result[name] = [{header[k]: value for k, value in row.items() if k in header}
                            for row in rows[1:] if row.get(0) not in (None, "")]
    return result


def assemble_ybus(tables):
    """Pure small-fixture-compatible assembly using the frozen bus1-tap/PQZ rules."""
    buses = tables["Bus"]
    ids = [int(b["idx"]) for b in buses]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate bus identity")
    order = {value: i for i, value in enumerate(ids)}
    bus = {int(b["idx"]): b for b in buses}
    Y = np.zeros((len(ids), len(ids)), complex)
    for line in tables["Line"]:
        f, t = int(line["bus1"]), int(line["bus2"])
        i, j = order[f], order[t]
        if line["Vn2"] != bus[t]["Vn"] or line["Vn1"] <= 0 or line["Sn"] <= 0:
            raise ValueError("line voltage or power base")
        zratio = (line["Vn1"]**2/line["Sn"])/(bus[f]["Vn"]**2/100)
        yratio = 1/zratio
        u, tap = line["u"], line["tap"]
        if tap <= 0:
            raise ValueError("nonpositive tap")
        ys = u/(complex(line["r"], line["x"])*zratio)
        yh = u*complex(line["g"], line["b"])*yratio/2
        y1 = u*complex(line["g1"], line["b1"])*yratio
        y2 = u*complex(line["g2"], line["b2"])*yratio
        m = tap*np.exp(1j*line["phi"])
        Y[i, i] += (ys+y1+yh)/tap**2
        Y[i, j] -= ys/m.conjugate()
        Y[j, i] -= ys/m
        Y[j, j] += ys+y2+yh
    for shunt in tables["Shunt"]:
        b = int(shunt["bus"])
        ratio = (bus[b]["Vn"]**2/100)/(shunt["Vn"]**2/shunt["Sn"])
        Y[order[b], order[b]] += shunt["u"]*complex(shunt["g"], shunt["b"])*ratio
    for load in tables["PQ"]:
        b = int(load["bus"])
        v = bus[b]["v0"]
        if v <= 0:
            raise ValueError("nonpositive PQ voltage anchor")
        Y[order[b], order[b]] += load["u"]*complex(load["p0"], -load["q0"])/v**2
    if not np.isfinite(Y).all():
        raise ValueError("nonfinite Ybus")
    return Y, ids


def schur_reference(Y, ids, retained):
    """Exactly one SVD and solve; no inverse, pseudoinverse, or regularization."""
    index = {b: i for i, b in enumerate(ids)}
    ri = [index[b] for b in retained]
    ei = [i for i, b in enumerate(ids) if b not in retained]
    EE, ER = Y[np.ix_(ei, ei)], Y[np.ix_(ei, ri)]
    RE, RR = Y[np.ix_(ri, ei)], Y[np.ix_(ri, ri)]
    singular = np.linalg.svd(EE, compute_uv=False)
    if not len(singular) or singular[-1] == 0:
        return _fail("singular_eliminated_block", singular_values=singular.tolist())
    try:
        T = np.linalg.solve(EE, ER)
    except np.linalg.LinAlgError:
        return _fail("elimination_solve_failed", singular_values=singular.tolist())
    reduced = RR-RE@T
    solve_res = np.linalg.norm(EE@T-ER)/max(1., np.linalg.norm(ER))
    embedding = np.zeros((len(ids), len(ri)), complex)
    embedding[ri] = np.eye(len(ri))
    embedding[ei] = -T
    relation = Y@embedding
    expected = np.zeros_like(relation)
    expected[ri] = reduced
    schur_res = np.linalg.norm(relation-expected)/max(1., np.linalg.norm(expected))
    symmetry = np.linalg.norm(Y-Y.T)/max(1., np.linalg.norm(Y))
    details = dict(singular_values=singular.tolist(), condition_number=float(singular[0]/singular[-1]),
                   solve_residual=float(solve_res), schur_residual=float(schur_res),
                   transpose_symmetry_error=float(symmetry),
                   reduced_transpose_symmetry_error=float(np.linalg.norm(reduced-reduced.T)/max(1., np.linalg.norm(reduced))),
                   residual_denominators="max(1,Frobenius(ER)), max(1,Frobenius(expected)), max(1,Frobenius(Y))",
                   svd_count=1, solve_count=1)
    if not np.isfinite(reduced).all() or solve_res > 1e-10 or schur_res > 1e-10 or symmetry > 1e-12 or details["reduced_transpose_symmetry_error"] > 1e-12:
        return _fail("Kron_qualification_failed", **details)
    return dict(status="qualified", Yred_real=reduced.real.tolist(), Yred_imag=reduced.imag.tolist(),
                retained_bus_ids=list(retained), **details)


def kron_reference(workbook_path):
    try:
        tables = read_case_workbook(workbook_path)
        if [len(tables[k]) for k in ("Bus", "Line", "Shunt", "PQ", "GENROU")] != [39, 46, 2, 19, 10]:
            return _fail("case_table_counts")
        if [int(r["bus"]) for r in tables["GENROU"]] != list(range(30, 40)):
            return _fail("generator_order")
        Y, ids = assemble_ybus(tables)
        if ids != list(range(1, 40)) or any(row["phi"] != 0 for row in tables["Line"]):
            return _fail("case_bus_order_or_phase")
        result = schur_reference(Y, ids, list(range(30, 40)))
        return dict(result, reference="bus_terminal_line_shunt_pqz_casev0_s100_v1", case_sha256=CASE_SHA)
    except (ValueError, KeyError, OSError, zipfile.BadZipFile, ET.ParseError, ZeroDivisionError) as exc:
        return _fail("Kron_source_or_assembly_failed", detail=str(exc))


def graph_correspondence(kron, functional, delta_star):
    if kron.get("status") != "qualified":
        return _fail("Kron_unqualified")
    Y = np.array(kron["Yred_real"])+1j*np.array(kron["Yred_imag"])
    k = functional.effective_edge_couplings(np.zeros(functional.n_nodes))
    y = np.array([abs(Y[i, j]) for i, j in functional.edges])
    correlation = None if np.ptp(k) == 0 or np.ptp(y) == 0 else float(spearmanr(k, y).statistic)
    p = _parameters(functional, delta_star)
    edge_rows = [dict(nodes=[i, j], exact_K=str(p["K"][e]), exact_beta=str(p["beta"][e]),
                      raw_supported=bool(p["K"][e] > 0),
                      reference_angle_stiffness=float(k[e]*np.cos(float(p["phi"][e]))),
                      admittance_magnitude=float(y[e]))
                 for e, (i, j) in enumerate(functional.edges)]
    return dict(status="reported", spearman=correlation, edges=edge_rows,
                null_graph=bool(all(k == 0 for k in p["K"])),
                admittance_weights=(y/y.sum()).tolist() if y.sum() else None,
                effective_weights=(k/k.sum()).tolist() if k.sum() else None,
                scope="terminal admittance versus effective angle coupling; report-only")


def _pole_distance(a, b):
    return abs(a[:, None]-b[None, :])/np.maximum(1., abs(b[None, :]))


def _pair_error(a, b):
    if len(a) != len(b):
        return float("inf")
    if not len(a):
        return 0.
    distance = _pole_distance(a, b)
    rows, cols = linear_sum_assignment(distance)
    return float(np.max(distance[rows, cols]))


def _mode_record(pole, shape, index):
    norm = float(np.linalg.norm(shape))
    if norm <= np.finfo(float).eps or not np.isfinite(shape).all():
        return None
    vector = shape/norm
    return dict(index=int(index), real=float(pole.real), imag=float(pole.imag),
                frequency_hz=float(abs(pole.imag)/(2*np.pi)),
                damping_ratio=float(-pole.real/abs(pole)), decay_rate=float(pole.real),
                strength=norm, shape_real=vector.real.tolist(), shape_imag=vector.imag.tolist())


def descriptor_modes(A, E, em_indices, omega_indices, mass, omega_s, accepted_poles):
    """Two QZ calls on the supplied pencil. Small fixtures only before release."""
    homogeneous, left, right = eig(A, E, left=True, homogeneous_eigvals=True)
    sums = np.sum(abs(homogeneous), axis=0)
    finite = abs(homogeneous[1]) > 1e-10*sums
    if np.any(sums < 1e-10):
        return _fail("indeterminate_descriptor_pair", qz_count=1)
    values = homogeneous[0, finite]/homogeneous[1, finite]
    right, left = right[:, finite], left[:, finite]
    denominator = (np.linalg.norm(A)+abs(values)*np.linalg.norm(E))*np.linalg.norm(right, axis=0)
    residuals = np.linalg.norm(A@right-(E@right)*values, axis=0)/denominator
    scale = np.maximum(np.max(abs(A)+abs(E), axis=1), 1e-10)
    scaled = eig(A/scale[:, None], E/scale[:, None], right=False, homogeneous_eigvals=True)
    scaled_sum = np.sum(abs(scaled), axis=0)
    other_finite = abs(scaled[1]) > 1e-10*scaled_sum
    other = scaled[0, other_finite]/scaled[1, other_finite]
    cross_error = _pair_error(values, other)
    checks = dict(finite_modes=len(values), infinite_modes=int(sum(~finite)),
                  indeterminate_pairs=0, row_equilibration_eigen_error=cross_error,
                  max_eigen_residual=float(np.max(residuals)),
                  positive_modes=int(sum(values.real > 1e-6)),
                  zero_modes=int(sum(abs(values) <= 1e-6)), qz_count=2)
    if np.any(scaled_sum < 1e-10) or not np.isfinite(residuals).all() or checks["max_eigen_residual"] >= 1e-8 or cross_error >= 1e-6:
        return _fail("descriptor_numerical_qualification", checks=checks)
    weights = left.conj().T@E
    overlaps = np.einsum("ij,ji->i", weights, right)
    modes = []
    unidentifiable = []
    for j, pole in enumerate(values):
        if abs(overlaps[j]) < 1e-12:
            unidentifiable.append(j)
            continue
        participation = abs(right[:, j]*weights[j]/overlaps[j])
        participation /= participation.sum()
        em = float(participation[em_indices].sum())
        if pole.imag > 1e-6 and em >= .1:
            shape = np.sqrt(mass)*omega_s*right[omega_indices, j]
            record = _mode_record(pole, shape, j)
            if record is None:
                return _fail("accepted_FOM_output_unobservable", checks=checks)
            modes.append(dict(record, electromechanical_participation=em))
    poles = np.array([complex(r["real"], r["imag"]) for r in modes])
    accepted_error = _pair_error(np.asarray(accepted_poles), poles)
    checks.update(accepted_pole_error=accepted_error, nonidentifiable_indices=unidentifiable)
    if accepted_error >= 1e-6:
        return _fail("accepted_mode_identity_changed", checks=checks)
    return dict(status="qualified", modes=modes, checks=checks,
                all_poles=[[float(v.real), float(v.imag)] for v in values],
                speed_units="canonical rad/s", weight="diag(sqrt(M)), all omega retained")


def fom_modes(descriptor_path, evidence_path, accepted_path, mass, *, expected_identity):
    """Hash-bound existing 699-dimensional reference, never ANDES or new data."""
    expected = expected_identity
    try:
        if [_sha(p) for p in (descriptor_path, evidence_path, accepted_path)] != [DESCRIPTOR_SHA, EVIDENCE_SHA, ACCEPTED_SHA]:
            return _fail("FOM_source_hash_mismatch")
        source = json.loads(Path(evidence_path).read_text(encoding="utf-8"))
        accepted = json.loads(Path(accepted_path).read_text(encoding="utf-8"))
        for key, value in (("andes", "2.0.0"), ("case_sha256", CASE_SHA), ("physical_signature", SIGNATURE)):
            if source.get(key) != value or expected.get(key) != value:
                return _fail("FOM_physical_identity_mismatch", field=key)
        if expected.get("generator_order") != [f"GENROU_{i}" for i in range(1, 11)]:
            return _fail("fit297_generator_order_unverified")
        if expected.get("equilibrium_identity_verified") is not True or not expected.get("equilibrium_verification_reference"):
            return _fail("fit297_equilibrium_identity_unverified")
        for key in ("equilibrium_state_error", "equilibrium_algebraic_error"):
            value = expected.get(key)
            if value is None or not np.isfinite(value) or abs(value) > 1e-8 or source[key] > 1e-8:
                return _fail("FOM_equilibrium_mismatch", field=key)
        omega_s = expected.get("omega_s")
        mass = np.asarray(mass, float)
        if mass.shape != (10,) or not np.isfinite(mass).all() or np.any(mass <= 0) or omega_s != 2*np.pi*60:
            return _fail("FOM_output_units_or_mass")
        with np.load(descriptor_path, allow_pickle=False) as data:
            names = data["state_names"].tolist()
            correct = [f"{kind} GENROU {i}" for kind in ("delta", "omega") for i in range(1, 11)]
            if names[:20] != correct:
                return _fail("descriptor_generator_order")
            A = np.block([[data["fx"], data["fy"]], [data["gx"], data["gy"]]])
            E = np.diag(np.r_[data["time_constants"], np.zeros(len(data["gy"]))])
        if A.shape != (699, 699) or not np.isfinite(A).all() or not np.isfinite(E).all():
            return _fail("descriptor_shape_or_finiteness")
        reference = accepted["fullorder_electromechanical_modes"]
        if len(reference) != 13:
            return _fail("accepted_reference_count")
        accepted_poles = np.array([complex(r["real"], r["imag"]) for r in reference])
        result = descriptor_modes(A, E, list(range(20)), list(range(10, 20)), mass, omega_s, accepted_poles)
        if result["status"] == "qualified":
            checks = result["checks"]
            original_poles = np.array([complex(r["real"], r["imag"]) for r in accepted["fullorder_all_modes"]])
            current_poles = np.array([complex(*v) for v in result["all_poles"]])
            checks["all_accepted_pole_error"] = _pair_error(original_poles, current_poles)
            if checks["all_accepted_pole_error"] >= 1e-6:
                return _fail("descriptor_accepted_spectrum_changed", checks=checks)
            for key, value in (("finite_modes", 150), ("infinite_modes", 549), ("zero_modes", 1), ("positive_modes", 0)):
                if checks[key] != value or accepted["descriptor_checks"][key] != value:
                    return _fail("descriptor_accepted_counts_changed", checks=checks)
        return result
    except (ValueError, KeyError, TypeError, OSError, np.linalg.LinAlgError) as exc:
        return _fail("FOM_reference_failed", detail=str(exc))


def quotient_basis(n):
    """Fixed Helmert basis; floating basis is only for numerical L3, never L2."""
    Q = np.zeros((n, n-1))
    for j in range(n-1):
        Q[:j+1, j] = 1/np.sqrt((j+1)*(j+2))
        Q[j+1, j] = -(j+1)/np.sqrt((j+1)*(j+2))
    if np.max(abs(Q.T@Q-np.eye(n-1))) > 1e-12 or np.max(abs(Q.sum(axis=0))) > 1e-12:
        raise ValueError("quotient basis qualification")
    return Q


def _central_jacobian(function, x, factor):
    columns = []
    for j in range(len(x)):
        h = factor*max(1., abs(x[j]))
        d = np.zeros_like(x)
        d[j] = h
        columns.append((function(x+d)-function(x-d))/(2*h))
    J = np.column_stack(columns)
    if not np.isfinite(J).all():
        raise ValueError("nonfinite finite-difference Jacobian")
    return J


def _rom_candidates(J, observation):
    values, vectors = np.linalg.eig(J)
    residual = np.linalg.norm(J@vectors-vectors*values, axis=0)/(
        (np.linalg.norm(J)+abs(values))*np.linalg.norm(vectors, axis=0))
    modes, excluded = [], []
    for j, pole in enumerate(values):
        frequency = abs(pole.imag)/(2*np.pi)
        reason = None
        if pole.imag <= 1e-10:
            reason = "nonpositive_imaginary"
        elif frequency < .01 or frequency > 10:
            reason = "outside_fixed_frequency_band"
        elif abs(pole) <= np.finfo(float).eps:
            reason = "zero_pole"
        record = None if reason else _mode_record(pole, observation@vectors[:, j], j)
        if reason is None and record is None:
            reason = "unobservable_or_nonfinite_omega"
        if reason:
            excluded.append(dict(index=j, real=float(pole.real), imag=float(pole.imag), reason=reason))
        else:
            modes.append(record)
    return modes, excluded, float(np.max(residual)) if np.isfinite(residual).all() else float("inf")


def rom_modes(functional, delta_star, certificate=None):
    """Bounded equilibrium search and two 49D grids; only common theta removed."""
    f, n = functional, functional.n_nodes
    counters = dict(newton_iterations=0, gradient_calls=0, rhs_calls=0, eig_count=0)
    try:
        Q = quotient_basis(n)
        T = block_diag(Q, np.eye(n), np.eye(n), np.eye(n), np.eye(n))
        theta = np.asarray(delta_star, float)
        theta = theta-theta.mean()
        center = np.r_[theta, np.zeros(4*n)]
        y = np.zeros(2*n-1)
        R = float(F(certificate["selected"]["R"])) if certificate and certificate.get("status") == "certified" else 1.
        def state(v):
            z = center.copy()
            z[:n] += Q@v[:n-1]
            z[3*n:4*n] = v[n-1:]
            return z
        def gradient(v):
            counters["gradient_calls"] += 1
            gt, _, _, ga, _ = f.gradients(state(v))
            return np.r_[Q.T@gt, ga]
        def rhs(z):
            counters["rhs_calls"] += 1
            return f.rhs(z)
        for _ in range(20):
            grad = gradient(y)
            if np.max(abs(rhs(state(y)))) <= 1e-10:
                break
            counters["newton_iterations"] += 1
            J = _central_jacobian(gradient, y, 1e-6)
            direction = np.linalg.solve(J, -grad)
            accepted = False
            for j in range(10):
                proposal = y+direction/2**j
                if np.linalg.norm(proposal) > R:
                    continue
                if certificate and certificate.get("status") == "certified" and exact_quotient_norm_squared(state(proposal), certificate["center_theta"]) >= F(certificate["selected"]["R"])**2:
                    continue
                if np.linalg.norm(gradient(proposal)) < np.linalg.norm(grad):
                    y, accepted = proposal, True
                    break
            if not accepted:
                return _fail("Newton_no_descent", counters=counters)
        equilibrium = state(y)
        residual = float(np.max(abs(rhs(equilibrium))))
        if not np.isfinite(residual) or residual > 1e-10:
            return _fail("ROM_equilibrium_unqualified", equilibrium_residual=residual, counters=counters)
        observation = np.zeros((n, 5*n))
        observation[:, n:2*n] = np.diag(np.sqrt(f.mass))
        reduced_rhs = lambda v: T.T@rhs(equilibrium+T@v)
        # Steps are in fixed reduced state coordinates at the equilibrium.
        coordinate = T.T@equilibrium
        def grid(factor):
            columns = []
            for j in range(5*n-1):
                h = factor*max(1., abs(coordinate[j]))
                d = np.zeros(5*n-1)
                d[j] = h
                columns.append((reduced_rhs(d)-reduced_rhs(-d))/(2*h))
            return np.column_stack(columns)
        main, fine = grid(1e-6), grid(5e-7)
        difference = float(np.linalg.norm(main-fine)/max(1., np.linalg.norm(fine)))
        if not np.isfinite(main).all() or not np.isfinite(fine).all() or difference >= 1e-6:
            return _fail("ROM_Jacobian_grid_unresolved", jacobian_difference=difference, counters=counters)
        modes_main, excluded_main, error_main = _rom_candidates(main, observation@T)
        modes_fine, excluded_fine, error_fine = _rom_candidates(fine, observation@T)
        counters["eig_count"] = 2
        poles_main = np.array([complex(r["real"], r["imag"]) for r in modes_main])
        poles_fine = np.array([complex(r["real"], r["imag"]) for r in modes_fine])
        pole_error = _pair_error(poles_main, poles_fine)
        all_main = modes_main+excluded_main
        all_fine = modes_fine+excluded_fine
        dm = np.array([complex(r["real"], r["imag"]) for r in all_main])
        df = np.array([complex(r["real"], r["imag"]) for r in all_fine])
        rr, cc = linear_sum_assignment(_pole_distance(dm, df))
        boundary_changed = any((i < len(modes_main)) != (j < len(modes_fine)) for i, j in zip(rr, cc, strict=True))
        if boundary_changed:
            return _fail("ROM_candidate_boundary_membership_changed", counters=counters)
        if max(error_main, error_fine) >= 1e-8 or pole_error >= 1e-6:
            return _fail("ROM_modes_grid_unresolved", pole_difference=pole_error, counters=counters)
        return dict(status="qualified", modes=modes_fine, excluded=excluded_fine,
                    main_excluded=excluded_main, equilibrium=equilibrium.tolist(),
                    equilibrium_residual=residual, jacobian_difference=difference,
                    pole_difference=pole_error, eigen_residuals=[error_main, error_fine],
                    dimension=5*n-1, retained_omega=n, counters=counters)
    except (ValueError, TypeError, OverflowError, FloatingPointError, np.linalg.LinAlgError) as exc:
        return _fail("ROM_modal_analysis_failed", detail=str(exc), counters=counters)


def match_modes(fom, rom):
    if fom.get("status") != "qualified" or rom.get("status") != "qualified":
        return _fail("unqualified_modal_input")
    keys = ("frequency_hz", "damping_ratio", "decay_rate", "strength", "shape_real", "shape_imag")
    ref = [DmdMode(**{k: row[k] for k in keys}) for row in fom["modes"]]
    cand = [DmdMode(**{k: row[k] for k in keys}) for row in rom["modes"]]
    matches = match_dmd_modes(ref, cand, max_relative_frequency_error=float("inf"))
    # Retain identities from the same full-cost assignment used by the shared helper.
    if ref and cand:
        cost = np.array([[abs(c.frequency_hz-r.frequency_hz)/max(r.frequency_hz, np.finfo(float).eps)
                          +1-modal_assurance_criterion(np.array(r.shape_real)+1j*np.array(r.shape_imag),
                                                       np.array(c.shape_real)+1j*np.array(c.shape_imag))
                          for c in cand] for r in ref])
        rows, cols = linear_sum_assignment(cost)
    else:
        rows, cols = [], []
    for match, i, j in zip(matches, rows, cols, strict=True):
        match.update(reference_index=int(i), candidate_index=int(j))
    return dict(status="reported", matches=matches,
                unmatched_reference=[i for i in range(len(ref)) if i not in rows],
                unmatched_candidate=[i for i in range(len(cand)) if i not in cols],
                scope="all assigned pairs, report-only; no MAC or frequency-error selection")
