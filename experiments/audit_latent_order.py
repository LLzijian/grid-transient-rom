"""Loop0059 coordinate-invariant audit of the frozen unwhitened r=3 model.

This is a diagnostic only.  It reads explicit TRAIN IDs and does not enumerate
or open TEST payloads. ``--synthetic`` exercises numerical contracts without
loading a project artifact.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from scipy.integrate import quad_vec
from scipy.linalg import expm

from kiki.data.private_dataset import canonical_arrays
from kiki.models.private_pm import MechanicalROM
from kiki.physics import coi_project

ROOTS = (Path("private_data/ieee39_postfault"), Path("private_data/ieee39_expansion"))
BASE = Path("private_data/task2_m3_0020/model.npz")
NETWORK = Path("private_data/task2_electrical_ports_0048/physical_network.npz")
MODEL = Path("private_data/task2_joint_0056/model.pt")
NORMALIZATION = Path("private_data/task2_joint_0056/normalization.npz")
RANK_TOLS = (1e-12, 1e-10, 1e-8)
FD_TOL = 1e-7
SIM_TOL = 1e-8


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_state(path):
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise TypeError("expected tensor state_dict")
    return {key: np.asarray(value.detach().cpu().numpy() if torch.is_tensor(value) else value, dtype=float)
            for key, value in raw.items()}


def read_train_events(base):
    """Read only IDs explicitly listed in each split's TRAIN list."""
    events, roots_meta = [], {}
    for root in ROOTS:
        splits = json.loads((root / "splits.json").read_text())
        train_ids = list(splits["train"])
        rows = {row["event_id"]: row for row in csv.DictReader((root / "dataset_index.csv").open())}
        roots_meta[str(root)] = {"splits_sha256": _sha(root / "splits.json"),
                                 "index_sha256": _sha(root / "dataset_index.csv"),
                                 "train_count_declared": len(train_ids)}
        with h5py.File(root / "dataset.h5", "r") as handle:
            equilibrium = {key: handle["equilibrium"][key][:] for key in ("delta", "omega", "Pm", "Pe")}
            for event_id in train_ids:
                if float(rows[event_id]["trajectory_end_s"]) <= .1 + 1e-9:
                    continue
                raw = handle["events"][event_id]
                arrays = {key: raw[key][:111] for key in ("t", "delta", "omega", "Pm", "Pe")}
                event = canonical_arrays(arrays, equilibrium, handle["inertia"][:], handle.attrs["omega_s"])
                x = np.column_stack((event["delta_sync"], event["omega_sync"], event["Pm_raw"] - base.pm_star))
                if len(x) > 11:
                    events.append({"id": event_id, "x": x, "group": rows[event_id]["fault_bus"],
                                   "label": rows[event_id]["label"]})
    return events, roots_meta


def _effective_xi(state):
    return np.asarray(state["xi"]) * np.asarray(state.get("support", np.ones_like(state["xi"])))


def _assert_sources(state, norm, base, network):
    """Fail closed if the saved JointROM buffers are not the declared M3 source."""
    for name in ("mass", "incidence", "delta_star", "coupling", "damping", "rate", "gain"):
        if name not in state or not np.array_equal(np.asarray(state[name]), np.asarray(getattr(base, name))):
            raise AssertionError(f"saved JointROM buffer {name} disagrees with M3")
    active = np.flatnonzero(np.asarray(base.coupling) > 0)
    if "angle_incidence" not in state or not np.array_equal(state["angle_incidence"], base.incidence[:, active]):
        raise AssertionError("saved active incidence disagrees with M3 coupling support")
    if "feature_scale" not in state or not np.array_equal(state["feature_scale"], norm["feature_scale"]):
        raise AssertionError("saved feature_scale disagrees with normalization")
    if "input_scale" not in state or not np.array_equal(state["input_scale"], norm["input_scale"]):
        raise AssertionError("saved input_scale disagrees with normalization")
    port_ids = np.asarray(norm["port_ids"], dtype=int)
    expected_ports = np.asarray(network["port_incidence"])[:, port_ids]
    if "ports" not in state or not np.array_equal(state["ports"], expected_ports):
        raise AssertionError("saved ports disagree with network / normalization port_ids")
    return active, expected_ports


def _physical_features(x, base, feature_scale):
    active = np.flatnonzero(np.asarray(base.coupling) > 0)
    incidence = np.asarray(base.incidence)[:, active]
    phase = np.asarray(base.delta_star) @ incidence
    x = np.asarray(x, dtype=float)
    eta = x[..., :10] @ incidence + phase
    physical = np.concatenate((x[..., 10:20], x[..., 20:30],
                               np.sin(eta) - np.sin(phase), np.cos(eta) - np.cos(phase)), axis=-1)
    return physical / np.asarray(feature_scale)


def _q_features(x, q, base, feature_scale):
    x, q = np.asarray(x, dtype=float), np.asarray(q, dtype=float)
    return np.concatenate((np.ones(x.shape[:-1] + (1,)), q, _physical_features(x, base, feature_scale)), axis=-1)


def qdot_feature(state, norm, base, x, q):
    """Frozen latent field used for independent central differences and g(0)."""
    return _q_features(x, q, base, norm["feature_scale"]) @ _effective_xi(state)


def build_matrices(state, norm, base, network):
    active, ports = _assert_sources(state, norm, base, network)
    xi, rank = _effective_xi(state), _effective_xi(state).shape[1]
    incidence, feature_scale = np.asarray(base.incidence)[:, active], np.asarray(norm["feature_scale"])
    nline = len(active)
    iomega, ip, isin, icos = 1 + rank, 1 + rank + 10, 1 + rank + 20, 1 + rank + 20 + nline
    Aq = xi[1:1 + rank].T
    B = np.zeros((rank, 30), dtype=float)
    B[:, 10:20], B[:, 20:30] = xi[iomega:ip].T / feature_scale[:10], xi[ip:isin].T / feature_scale[10:20]
    phase = np.asarray(base.delta_star) @ incidence
    B[:, :10] = (xi[isin:icos].T @ np.diag(np.cos(phase) / feature_scale[20:20 + nline]) @ incidence.T
                 + xi[icos:icos + nline].T @ np.diag(-np.sin(phase) / feature_scale[20 + nline:20 + 2 * nline]) @ incidence.T)
    c, amplitude = np.asarray(state["c"]), np.asarray(state["amplitude"])
    c_norm = np.linalg.norm(c, axis=0, keepdims=True)
    if np.any(c_norm == 0):
        raise AssertionError("saved latent port direction has a zero column")
    c_unit = ports @ (c / c_norm)
    return Aq, B, c_unit * amplitude[None, :], {"active": active, "feature_scale": feature_scale, "C_unit": c_unit}


def representative_B(state, norm, base, x):
    xi, rank = _effective_xi(state), _effective_xi(state).shape[1]
    active = np.flatnonzero(np.asarray(base.coupling) > 0)
    incidence, nline, feature_scale = np.asarray(base.incidence)[:, active], len(active), np.asarray(norm["feature_scale"])
    eta = np.asarray(x)[:10] @ incidence + np.asarray(base.delta_star) @ incidence
    iomega, ip, isin, icos = 1 + rank, 1 + rank + 10, 1 + rank + 20, 1 + rank + 20 + nline
    result = np.zeros((rank, 30), dtype=float)
    result[:, 10:20], result[:, 20:30] = xi[iomega:ip].T / feature_scale[:10], xi[ip:isin].T / feature_scale[10:20]
    result[:, :10] = (xi[isin:icos].T @ np.diag(np.cos(eta) / feature_scale[20:20 + nline]) @ incidence.T
                      + xi[icos:icos + nline].T @ np.diag(-np.sin(eta) / feature_scale[20 + nline:20 + 2 * nline]) @ incidence.T)
    return result


def fd_B_check(state, norm, base, x, h=1e-6):
    x, rank = np.asarray(x, dtype=float), _effective_xi(state).shape[1]
    numerical = np.column_stack([(qdot_feature(state, norm, base, x + np.eye(30)[j] * h, np.zeros(rank))
                                  - qdot_feature(state, norm, base, x - np.eye(30)[j] * h, np.zeros(rank))) / (2 * h)
                                 for j in range(30)])
    return float(np.max(np.abs(numerical - representative_B(state, norm, base, x))))


def fd_Aq_check(state, norm, base, x, h=1e-6):
    rank = _effective_xi(state).shape[1]
    numerical = np.column_stack([(qdot_feature(state, norm, base, x, np.eye(rank)[j] * h)
                                  - qdot_feature(state, norm, base, x, -np.eye(rank)[j] * h)) / (2 * h)
                                 for j in range(rank)])
    return float(np.max(np.abs(numerical - _effective_xi(state)[1:1 + rank].T)))


def _matrix_diagnostics(matrix):
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    largest = singular_values[0] if len(singular_values) else 0.0
    singular = not len(singular_values) or largest == 0.0 or singular_values[-1] <= np.finfo(float).eps * largest
    return {"singular_values": singular_values.tolist(), "condition": None if singular else float(largest / singular_values[-1]),
            "numerically_singular_null_flag": bool(singular),
            "rank_rtol": {str(tol): int(np.sum(singular_values > largest * tol)) if largest else 0 for tol in RANK_TOLS}}


def _pbh(A, B, C, poles):
    eye = np.eye(A.shape[0], dtype=complex)
    return [{"pole": [float(pole.real), float(pole.imag)],
             "reachability": _matrix_diagnostics(np.hstack((pole * eye - A, B))),
             "observability": _matrix_diagnostics(np.vstack((pole * eye - A, C)))} for pole in poles]


def _complex_list(value):
    return [[float(z.real), float(z.imag)] for z in np.asarray(value).ravel()]


def _modal_blocks(A):
    """Build invariant real/conjugate/repeated projectors; never invent eigenmodes."""
    poles, V = np.linalg.eig(A)
    residual, condition = float(np.linalg.norm(A @ V - V * poles[None, :], ord=np.inf)), float(np.linalg.cond(V))
    if not np.isfinite(condition) or condition > 1e12 or residual > 1e-7:
        return poles, None, None, [], {"defective_or_ill_conditioned": True, "eigenvector_condition": condition if np.isfinite(condition) else None,
                                        "eigen_residual_inf": residual, "warning": "modal projectors unresolved"}
    groups, unused, scale = [], set(range(len(poles))), max(1.0, float(np.max(np.abs(poles))))
    while unused:
        index, pole = min(unused), poles[min(unused)]
        repeated = [j for j in unused if abs(poles[j] - pole) <= 1e-8 * scale]
        if len(repeated) > 1:
            indices, kind = sorted(repeated), "repeated_block"
        elif abs(pole.imag) > 1e-10 * scale:
            mates = [j for j in unused if j != index and abs(poles[j] - np.conj(pole)) <= 1e-8 * scale]
            if len(mates) != 1:
                return poles, None, None, [], {"defective_or_ill_conditioned": True, "eigenvector_condition": condition,
                                                "eigen_residual_inf": residual, "warning": "unpaired complex pole"}
            indices, kind = sorted((index, mates[0])), "conjugate_block"
        else:
            indices, kind = [index], "simple_real"
        unused.difference_update(indices)
        groups.append({"indices": indices, "kind": kind})
    W = np.linalg.inv(V)
    for group in groups:
        indices = group["indices"]
        group["poles"] = [[float(poles[i].real), float(poles[i].imag)] for i in indices]
        group["projector"] = np.real_if_close(V[:, indices] @ W[indices, :], tol=1000)
    return poles, V, W, groups, {"defective_or_ill_conditioned": False, "eigenvector_condition": condition,
                                  "eigen_residual_inf": residual}


def pole_report(A, B0, C):
    poles, V, W, blocks, report = _modal_blocks(A)
    report.update({"eigenvalues": _complex_list(poles), "decay_times_negative_real_only": [-1.0 / float(p.real) for p in poles if p.real < 0],
                   "frequencies_hz": [float(abs(p.imag) / (2 * np.pi)) for p in poles],
                   "pairwise_separation": [float(abs(poles[i] - poles[j])) for i in range(len(poles)) for j in range(i)],
                   "distinct_time_scales_are_not_a_minimal_order_test": True})
    if V is not None:
        for block in blocks:
            projector = block.pop("projector")
            block["reference_residue_norm"] = float(np.linalg.norm(C @ projector @ B0))
            block["projector_real_imag"] = _complex_list(projector)
    report["modal_blocks"] = blocks
    return poles, V, W, blocks, report


def _real_block_basis(poles, V, blocks):
    """Real full basis with conjugate pairs intact and repeated poles grouped."""
    columns, groups, kinds = [], [], []
    for block in blocks:
        start = len(columns)
        if block["kind"] == "conjugate_block":
            positive = next(index for index in block["indices"] if poles[index].imag > 0)
            vector = V[:, positive]
            columns.extend((np.real(vector), np.imag(vector)))
        else:
            columns.extend(np.real_if_close(V[:, index], tol=1000).real for index in block["indices"])
        groups.append((start, len(columns)))
        kinds.append(block["kind"])
    basis = np.column_stack(columns)
    if not np.isfinite(np.linalg.cond(basis)) or np.linalg.cond(basis) > 1e12:
        raise AssertionError("real modal block basis is ill conditioned")
    return basis, np.linalg.inv(basis), np.asarray(groups, dtype=int), np.asarray(kinds)


def _event_weighted_rms(event_arrays):
    return None if not event_arrays else float(np.sqrt(np.mean([np.mean(np.abs(values) ** 2) for values in event_arrays])))


def _projector(block):
    return np.asarray([z[0] + 1j * z[1] for z in block["projector_real_imag"]]).reshape(3, 3)


def modal_excitation(blocks, events, state, norm, base, C, trajectories):
    if not blocks:
        return [{"warning": "modal blocks unresolved; excitation not fabricated"}]
    rank = _effective_xi(state).shape[1]
    forcing = {event["id"]: qdot_feature(state, norm, base, event["x"][10:111], np.zeros((len(event["x"][10:111]), rank))) for event in events}
    reference = qdot_feature(state, norm, base, np.zeros(30), np.zeros(rank))
    initial, expected = {}, {f"{event['id']}_encoded_q": event["id"] for event in events}
    if trajectories is not None:
        for key, event_id in expected.items():
            try:
                q = np.asarray(trajectories[key])
            except KeyError as error:
                raise AssertionError(f"missing declared TRAIN initial latent: {key}") from error
            if q.ndim != 2 or q.shape[1] != rank or not len(q):
                raise AssertionError(f"invalid TRAIN encoded latent array: {key}")
            initial[event_id] = q[0]
    rows = []
    for block in blocks:
        projector = _projector(block)
        actual = [(C @ projector @ forcing[event["id"]].T).T for event in events]
        delta = [(C @ projector @ (forcing[event["id"]] - reference).T).T for event in events]
        row = {"kind": block["kind"], "poles": block["poles"], "forcing_rms": _event_weighted_rms(actual),
               "delta_from_reference_rms": _event_weighted_rms(delta), "initial_q_count": len(initial)}
        if initial:
            row["initial_output_rms"] = float(np.sqrt(np.mean([np.mean(np.abs(C @ projector @ q) ** 2) for q in initial.values()])))
        rows.append(row)
    return rows


def finite_gramians(A, B, C, horizon, Q=None):
    Q, P = B @ B.T if Q is None else np.asarray(Q), C.T @ C
    wc = quad_vec(lambda time: expm(A * time) @ Q @ expm(A.T * time), 0.0, horizon, epsabs=1e-10, epsrel=1e-10)[0]
    wo = quad_vec(lambda time: expm(A.T * time) @ P @ expm(A * time), 0.0, horizon, epsabs=1e-10, epsrel=1e-10)[0]
    wc, wo = (wc + wc.T) / 2, (wo + wo.T) / 2
    def psd_root(matrix):
        values, vectors = np.linalg.eigh(matrix)
        if np.min(values) < -1e-9 * max(1.0, np.max(np.abs(values))):
            raise AssertionError("finite Gramian is not PSD to numerical tolerance")
        return (vectors * np.sqrt(np.maximum(values, 0.0))) @ vectors.T
    root_c, root_o = psd_root(wc), psd_root(wo)
    hsv = np.sort(np.linalg.svd(root_o @ root_c, compute_uv=False))[::-1]
    if np.any(hsv < -1e-12):
        raise AssertionError("Hankel singular values must be nonnegative")
    U, singular_values, Vt = np.linalg.svd(root_o.T @ root_c)
    if not len(singular_values) or np.any(singular_values <= 1e-12 * max(singular_values[0], 1.0)):
        balanced = {"singular": True, "singular_values": singular_values.tolist(),
                    "warning": "rank-deficient finite Gramian; no inverse-balanced basis invented"}
    else:
        V, W = root_c @ Vt.T @ np.diag(singular_values ** -0.5), np.diag(singular_values ** -0.5) @ U.T @ root_o.T
        error = float(np.max(np.abs(W @ V - np.eye(len(singular_values)))))
        if error >= SIM_TOL:
            raise AssertionError(f"balanced inverse check failed: {error}")
        balanced = {"singular": False, "V": V.tolist(), "W": W.tolist(), "biorthogonality_error": error}
    return wc, wo, hsv, balanced


def similarity_check(A, B, C, Q):
    T = np.array([[2.0, .4, -.2], [0.0, .5, .3], [.1, 0.0, 1.5]])
    if A.shape != (3, 3):
        raise AssertionError("preregistered similarity matrix requires r=3")
    At, Bt, Ct = T @ A @ np.linalg.inv(T), T @ B, C @ np.linalg.inv(T)
    original_poles, transformed_poles = np.sort_complex(np.linalg.eigvals(A)), np.sort_complex(np.linalg.eigvals(At))
    pole_error = float(np.max(np.abs(original_poles - transformed_poles) / np.maximum(np.abs(original_poles), 1.0)))
    transfer_errors = {}
    for point in (1 + 1j, 3 + 2j, 10 + 3j):
        original = C @ np.linalg.solve(point * np.eye(3) - A, B)
        transformed = Ct @ np.linalg.solve(point * np.eye(3) - At, Bt)
        transfer_errors[str(point)] = float(np.max(np.abs(original - transformed)) / max(float(np.max(np.abs(original))), 1.0))
    hsv_errors, hsv_absolute_errors = {}, {}
    for horizon in (1.0, 2.0, 8.0):
        _, _, original, _ = finite_gramians(A, B, C, horizon, Q)
        _, _, transformed, _ = finite_gramians(At, Bt, Ct, horizon, T @ Q @ T.T)
        absolute = np.abs(original - transformed)
        floor = 1e-12 * max(float(np.max(np.abs(original))), np.finfo(float).tiny)
        hsv_absolute_errors[str(horizon)] = float(np.max(absolute))
        hsv_errors[str(horizon)] = float(np.max(absolute / np.maximum(np.abs(original), floor)))
    q0 = np.array([.2, -.3, .4])
    trajectory_errors = [float(np.max(np.abs(C @ expm(A * time) @ q0 - Ct @ expm(At * time) @ (T @ q0)))) for time in (0., 1., 2., 8.)]
    result = {"T": T.tolist(), "pole_relative_error": pole_error, "transfer_relative_errors": transfer_errors,
              "hsv_relative_errors": hsv_errors, "hsv_absolute_errors": hsv_absolute_errors,
              "physical_output_trajectory_max_abs_errors": trajectory_errors}
    if pole_error >= SIM_TOL or max(transfer_errors.values()) >= SIM_TOL or max(hsv_errors.values()) >= SIM_TOL:
        raise AssertionError(f"similarity contract failed: {result}")
    return result


def _representatives(events):
    return [{"event_id": event["id"], "index": index, "x": event["x"][index].tolist()}
            for event in events for index in (10, 30, 60, 110) if index < len(event["x"])]


def _assert_fd(errors):
    if max(errors) >= FD_TOL:
        raise AssertionError(f"analytic derivative / central FD max error {max(errors)} >= {FD_TOL}")


def audit(args):
    base, network = MechanicalROM.load(args.base), dict(np.load(args.network))
    state, norm = _load_state(args.model), dict(np.load(args.normalization))
    Aq, Bphysical, Cq, extra = build_matrices(state, norm, base, network)
    events, data_meta = read_train_events(base)
    reps = _representatives(events)
    first_second = [np.column_stack((coi_project(event["x"][10:111, :10], base.mass), event["x"][10:111, 10:])) for event in events]
    scales = np.maximum(np.sqrt(np.mean([np.mean(values ** 2, axis=0) for values in first_second], axis=0)), .01)
    Bscaled = Bphysical @ np.diag(scales)
    Bphysical_reps = np.stack([representative_B(state, norm, base, np.asarray(rep["x"])) for rep in reps])
    Bscaled_reps = Bphysical_reps * scales[None, None, :]
    per_event = [Bscaled_reps[[j for j, rep in enumerate(reps) if rep["event_id"] == event["id"]]] for event in events]
    Qpooled = np.mean([np.mean(item @ np.swapaxes(item, 1, 2), axis=0) for item in per_event], axis=0)
    poles, V, W, blocks, pole_info = pole_report(Aq, Bscaled, Cq)
    O = np.vstack((Cq, Cq @ Aq, Cq @ Aq @ Aq))
    Rphysical = np.hstack((Bphysical, Aq @ Bphysical, Aq @ Aq @ Bphysical))
    Rscaled = np.hstack((Bscaled, Aq @ Bscaled, Aq @ Aq @ Bscaled))
    rep_reports = []
    for rep, physical, scaled in zip(reps, Bphysical_reps, Bscaled_reps, strict=True):
        residues = [{"kind": block["kind"], "poles": block["poles"], "physical_residue_norm": float(np.linalg.norm(Cq @ _projector(block) @ physical)),
                     "train_scaled_residue_norm": float(np.linalg.norm(Cq @ _projector(block) @ scaled))} for block in blocks]
        rep_reports.append({**rep, "R_physical": _matrix_diagnostics(np.hstack((physical, Aq @ physical, Aq @ Aq @ physical))),
                            "R_train_scaled": _matrix_diagnostics(np.hstack((scaled, Aq @ scaled, Aq @ Aq @ scaled))),
                            "PBH_physical": _pbh(Aq, physical, Cq, poles), "PBH_train_scaled": _pbh(Aq, scaled, Cq, poles), "modal_residues": residues})
    trajectories = np.load(args.model.parent / "trajectories.npz") if (args.model.parent / "trajectories.npz").exists() else None
    fd_points = [np.zeros(30)] + [np.asarray(rep["x"]) for rep in reps[:3]]
    fd_b, fd_a = [fd_B_check(state, norm, base, point) for point in fd_points], [fd_Aq_check(state, norm, base, point) for point in fd_points]
    _assert_fd(fd_b + fd_a)
    rank = Aq.shape[0]
    g0_masked = qdot_feature(state, norm, base, np.zeros(30), np.zeros(rank))
    full_reference = np.r_[np.zeros(30), g0_masked]
    force_evidence = json.loads((args.model.parent / "evidence.json").read_text())
    force_error = float(np.max(np.abs(np.linalg.svd(Cq, compute_uv=False) - force_evidence["force_singular_values"])))
    if force_error >= 1e-10:
        raise AssertionError(f"saved force singular values mismatch: {force_error}")
    similarity = similarity_check(Aq, Bscaled, Cq, Qpooled)
    out = {"matrix_contract": "A=q Jacobian; Cq=electrical-power/torque correction; Bphysical=[delta rad, omega rad/s, p pu]; Bscaled=Bphysical diag(TRAIN event-balanced RMS)",
           "A": Aq.tolist(), "Bphysical": Bphysical.tolist(), "Bscaled": Bscaled.tolist(), "C_unit": extra["C_unit"].tolist(), "Cq": Cq.tolist(),
           "input_scale_event_balanced_rms_floor_0.01": scales.tolist(), "poles": pole_info,
           "observability": _matrix_diagnostics(O), "reachability_physical": _matrix_diagnostics(Rphysical), "reachability_train_scaled": _matrix_diagnostics(Rscaled),
           "PBH_reference_physical": _pbh(Aq, Bphysical, Cq, poles), "PBH_reference_train_scaled": _pbh(Aq, Bscaled, Cq, poles),
           "representatives": rep_reports, "reference_residual": {"g0_masked": g0_masked.tolist(), "Xi_const_effective": _effective_xi(state)[0].tolist(), "full_reference_vector": full_reference.tolist(),
           "full_reference_norm": float(np.linalg.norm(full_reference)), "is_equilibrium": bool(np.linalg.norm(full_reference) <= 1e-8)},
           "modal_excitation": modal_excitation(blocks, events, state, norm, base, Cq, trajectories), "analytic_B_central_fd_max_abs": fd_b,
           "analytic_Aq_qfeature_central_fd_max_abs": fd_a, "force_singular_values_vs_saved_evidence_max_abs": force_error, "train_events": len(events),
           "data": data_meta, "test_read": False, "pooled_Q_eventweighted_representative": Qpooled.tolist(), "similarity_check": similarity,
           "source_sha256": {"model": _sha(args.model), "normalization": _sha(args.normalization), "base": _sha(args.base), "network": _sha(args.network), "audit_code": _sha(Path(__file__))},
           "coordinate_warning": "Raw O/R spectra and conditions are coordinate/unit dependent; ranks, poles and transfer maps are invariant."}
    arrays = {"A": Aq, "Cq": Cq, "C_unit": extra["C_unit"], "Bphysical": Bphysical, "Bscaled": Bscaled, "O": O, "Rphysical": Rphysical, "Rscaled": Rscaled,
              "input_scale": scales, "B_representative_physical": Bphysical_reps, "B_representative_scaled": Bscaled_reps,
              "representative_event_ids": np.asarray([rep["event_id"] for rep in reps]), "representative_indices": np.asarray([rep["index"] for rep in reps], dtype=int), "Qpooled": Qpooled}
    if V is not None:
        block_V, block_W, block_groups, block_kinds = _real_block_basis(poles, V, blocks)
        arrays.update({"modal_real_block_V": block_V, "modal_real_block_W": block_W,
                       "modal_block_column_groups": block_groups, "modal_block_kinds": block_kinds})
        if all(block["kind"] == "simple_real" for block in blocks):
            arrays.update({"modal_V": np.real(V), "modal_W": np.real(W), "modal_eigenvalues": np.real(poles)})
    for horizon in (1., 2., 8.):
        wc, wo, hsv, balanced = finite_gramians(Aq, Bscaled, Cq, horizon, Qpooled)
        out[f"gramian_T{horizon:g}"] = {"Wc": wc.tolist(), "Wo": wo.tolist(), "HSV": hsv.tolist(), "balanced": balanced,
                                        "scope": "pooled local-linear TRAIN-weighted finite-horizon diagnostic"}
        arrays.update({f"Wc_T{horizon:g}": wc, f"Wo_T{horizon:g}": wo, f"HSV_T{horizon:g}": hsv})
        if horizon == 1. and not balanced["singular"]:
            Vb, Wb = np.asarray(balanced["V"]), np.asarray(balanced["W"])
            if Vb.shape != (3, 3) or Wb.shape != (3, 3):
                raise AssertionError("T=1 balanced matrices must be full 3x3 before rank-2 selection")
            arrays.update({"balanced_V_T1": Vb, "balanced_W_T1": Wb})
    np.savez_compressed(args.output / "matrices.npz", **arrays)
    (args.output / "evidence.json").write_text(json.dumps(out, indent=2, allow_nan=False), encoding="utf-8")


def _synthetic_base(nline=2):
    incidence = np.zeros((10, nline)); incidence[0, :], incidence[1, :] = 1., -1.
    return SimpleNamespace(mass=np.arange(1., 11.), incidence=incidence, delta_star=np.zeros(10), coupling=np.ones(nline), damping=np.ones(10), rate=np.ones(10), gain=np.ones(10))


def synthetic_checks():
    rng, base, rank = np.random.default_rng(59), _synthetic_base(), 3
    nline = base.incidence.shape[1]
    state = {"xi": rng.normal(size=(1 + rank + 20 + 2 * nline, rank)), "support": np.ones((1 + rank + 20 + 2 * nline, rank))}
    norm, point = {"feature_scale": np.exp(rng.normal(size=20 + 2 * nline))}, rng.normal(size=30)
    b_error, a_error = fd_B_check(state, norm, base, point), fd_Aq_check(state, norm, base, point)
    _assert_fd([b_error, a_error])
    weighting = _event_weighted_rms([np.array([[3.]]), np.array([[4.], [4.]])])
    if not np.isclose(weighting, np.sqrt(12.5)):
        raise AssertionError("event weighting is not sqrt(mean event mean-square)")
    singular = _matrix_diagnostics(np.diag([1., 0.]))
    if not singular["numerically_singular_null_flag"] or singular["condition"] is not None:
        raise AssertionError("rank-deficiency reporting failed")
    _, _, _, complex_blocks, _ = _modal_blocks(np.array([[-1., -2., 0.], [2., -1., 0.], [0., 0., -3.]]))
    _, _, _, repeated_blocks, _ = _modal_blocks(np.diag([-1., -1., -2.]))
    if sorted(block["kind"] for block in complex_blocks) != ["conjugate_block", "simple_real"] or "repeated_block" not in [block["kind"] for block in repeated_blocks]:
        raise AssertionError("complex/repeated modal block handling failed")
    B, C = np.eye(3), np.array([[1., -.5, .2], [0., 1., .4]])
    for A in (np.diag([-.2, -.7, -1.4]), np.diag([.2, -.7, -1.4])):
        wc, wo, hsv, balanced = finite_gramians(A, B, C, 2., B @ B.T)
        if not (np.isfinite(wc).all() and np.isfinite(wo).all() and np.all(hsv >= 0) and not balanced["singular"]):
            raise AssertionError("finite Gramian contract failed")
        similarity_check(A, B, C, B @ B.T)
    return {"feature_jacobian_fd_max_abs": b_error, "q_jacobian_fd_max_abs": a_error, "event_weighted_rms": weighting, "status": "passed"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--normalization", type=Path, default=NORMALIZATION)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--network", type=Path, default=NETWORK)
    parser.add_argument("--synthetic", action="store_true", help="run numerical contracts without project artifacts")
    args = parser.parse_args()
    if args.synthetic:
        if args.output is not None:
            raise ValueError("--synthetic does not write an artifact")
        print(json.dumps(synthetic_checks(), indent=2)); return
    if args.output is None:
        parser.error("--output is required unless --synthetic is used")
    args.output.mkdir(parents=True, exist_ok=False)
    audit(args)


if __name__ == "__main__":
    main()
