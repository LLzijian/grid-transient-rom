"""M0--M4 frozen p-port diagnostics. No optimizer and no ROM rollout entry point."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src")]
PRIVATE = ROOT / "private_data/p_port_diagnostic_20261031"
PUBLIC = ROOT / "results/p_port_diagnostic_20261031/public_diagnostic.json"
PREVIOUS = ROOT / "private_data/checkpoint_selection_20261030"
MODELS = ("A96", "step000016")
SCOPES = ("stable5", "stable8", "unstable_actual15")
SCHEMA = "p_port_diagnostic_20261031_v1"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ident(path):
    path = Path(path).resolve()
    return {"path": path.relative_to(ROOT.resolve()).as_posix(), "sha256": sha(path)}


def safe(x):
    if isinstance(x, dict):
        return {str(k): safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [safe(v) for v in x]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else None
    return x


def write_once(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(safe(payload), f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def phi12(x):
    """Stable entire functions expm1(x)/x and (expm1(x)-x)/x**2."""
    x = np.asarray(x, dtype=np.float64)
    small = np.abs(x) < 1e-4
    with np.errstate(divide="ignore", invalid="ignore"):
        first = np.expm1(x) / x
        second = (np.expm1(x) - x) / x**2
    p1 = sum(x**k / math.factorial(k + 1) for k in range(8))
    p2 = sum(x**k / math.factorial(k + 2) for k in range(8))
    return np.where(small, p1, first), np.where(small, p2, second)


def linear_filter(omega, p0, g, a, dt):
    """Exact p'=-g*w-a*p for piecewise-linear w, including signed dt."""
    omega = np.asarray(omega, dtype=np.float64)
    p0, g, a = (np.asarray(v, dtype=np.float64) for v in (p0, g, a))
    if omega.ndim != 2 or omega.shape[1:] != p0.shape or p0.shape != g.shape or g.shape != a.shape or not np.isfinite(dt) or dt == 0:
        raise ValueError("filter requires [time,machine], matching vectors and nonzero dt")
    if not all(np.isfinite(v).all() for v in (omega, p0, g, a)):
        raise ValueError("nonfinite filter input")
    e = np.exp(-a * dt)
    f1, f2 = phi12(-a * dt)
    out = np.empty_like(omega)
    out[0] = p0
    for k in range(len(omega) - 1):
        value = e * out[k] - g * dt * (omega[k] * f1 + (omega[k + 1] - omega[k]) * f2)
        out[k + 1] = np.where(a == 0, out[k] - g * dt * (omega[k] + omega[k+1]) / 2, value)
    return out


def right_endpoint_filter(omega, p0, g, a, dt):
    """Analytic constant physical-time right endpoint control; sensitivity only."""
    omega, p0, g, a = (np.asarray(v, dtype=np.float64) for v in (omega, p0, g, a))
    f1, _ = phi12(-a * dt)
    decay = np.exp(-a * dt)
    out = np.empty_like(omega)
    out[0] = p0
    for k in range(len(omega)-1):
        w = omega[k+1] if dt > 0 else omega[k]
        out[k+1] = decay*out[k] - g*dt*f1*w
    return out


def normalized_rms(value, pscale):
    return float(np.sqrt(np.mean((np.asarray(value) / np.asarray(pscale))**2)))


def self_closure_gate(r, s):
    return bool(r<=.001 and r<=max(1e-10,2*s))


def reverse_matrix(g, a, dt=-.01):
    """Loaded p_backward maps [terminal p, terminal omega, forward omega slope]."""
    g, a = np.asarray(g), np.asarray(a)
    f1, f2 = phi12(-a * dt)
    return np.concatenate((np.diag(np.exp(-a*dt)), np.diag(-g*dt*f1), np.diag(-g*dt*dt*f2)), axis=1)


def squared_decomposition(predicted_p, model_filter_p, truth_filter_p, truth_p, pscale):
    """Future rows only; retain three squares and all three signed cross terms."""
    arrays = [np.asarray(v, dtype=np.float64) for v in (predicted_p, model_filter_p, truth_filter_p, truth_p)]
    if not arrays or any(v.shape != arrays[0].shape for v in arrays) or not all(np.isfinite(v).all() for v in arrays):
        raise ValueError("finite aligned decomposition paths required")
    components = [arrays[0] - arrays[1], arrays[1] - arrays[2], arrays[2] - arrays[3]]
    total = arrays[0] - arrays[3]
    names = ("implementation", "omega_transmission", "fixed_truth_driven")
    squares = {name: np.mean(c * c, axis=0) for name, c in zip(names, components)}
    cross = {f"{names[i]}__{names[j]}": 2 * np.mean(components[i] * components[j], axis=0) for i, j in ((0, 1), (0, 2), (1, 2))}
    total_mse = np.mean(total * total, axis=0)
    rebuilt = sum(squares.values()) + sum(cross.values())
    residual = total_mse - rebuilt
    np.testing.assert_allclose(sum(components), total, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(rebuilt, total_mse, atol=1e-12, rtol=1e-12)
    scale = np.asarray(pscale, dtype=np.float64)**2
    return {"per_machine_mse_pu2": {**squares, **cross, "total": total_mse},
        "per_machine_normalized_mse": {key: value / scale for key, value in {**squares, **cross, "total": total_mse}.items()},
        "per_machine_rmse_pu": {name: np.sqrt(value) for name, value in {**squares, "total": total_mse}.items()},
        "identity_max_abs_pu2": float(np.max(np.abs(residual))), "future_rows": len(total),
        "note": "Cross terms are signed; squared terms are not mutually exclusive causal shares."}


def window_integrals(path, scheme="simpson", dt=.01):
    """Fixed disjoint 20ms windows [0,2],[2,4],...; no partial-window fill."""
    path = np.asarray(path, dtype=np.float64)
    count = (len(path) - 1) // 2
    first, middle, last = path[:2 * count:2], path[1:2 * count:2], path[2:2 * count + 1:2]
    if scheme == "simpson":
        integral = dt / 3 * (first + 4 * middle + last)
    elif scheme == "trapezoid":
        integral = dt / 2 * (first + 2 * middle + last)
    else:
        raise ValueError("registered Simpson/trapezoid only")
    return integral, last - first, (len(path) - 1) - 2 * count


def event_equal_svd(event_designs, event_residuals):
    """One column-RMS-scaled SVD; each event and its windows have equal weight."""
    valid = [(np.asarray(x, dtype=np.float64), np.asarray(b, dtype=np.float64)) for x, b in zip(event_designs, event_residuals) if len(x)]
    if not valid:
        return {"status": "unknown", "reason": "no_complete_windows", "rank": None, "correction_projection_coefficients": None}
    if len(valid) != len(event_designs) or len(event_designs) != len(event_residuals):
        return {"status": "unknown", "reason": "incomplete_event_windows_no_reweighting", "rank": None, "correction_projection_coefficients": None}
    if any(x.ndim != 2 or x.shape[1] != 2 or b.shape != (len(x),) or not np.isfinite(x).all() or not np.isfinite(b).all() for x, b in valid):
        raise ValueError("finite paired two-column designs required")
    event_count = len(valid)
    weights = np.concatenate([np.full(len(x), 1 / (event_count * len(x))) for x, _ in valid])
    x = np.concatenate([x for x, _ in valid])
    b = np.concatenate([b for _, b in valid])
    xw, bw = x * np.sqrt(weights[:, None]), b * np.sqrt(weights)
    scales = np.linalg.norm(xw, axis=0)
    nonzero = scales > 0
    scale = np.where(nonzero, scales, 1.)
    z = xw / scale
    u, singular, vt = np.linalg.svd(z, full_matrices=False)
    threshold = max(z.shape) * np.finfo(np.float64).eps * singular[0]
    keep = singular > threshold
    rank = int(keep.sum())
    coeff_scaled = vt[keep].T @ ((u[:, keep].T @ bw) / singular[keep])
    coefficients = coeff_scaled / scale
    projection = x @ coefficients
    residual = b - projection
    return {"status": "computed", "event_count": event_count, "window_count": len(x), "column_weighted_rms": scales,
        "singular_values_scaled": singular, "rank_threshold": threshold, "rank": rank,
        "condition_scaled": float(singular[0] / singular[-1]) if rank == 2 else None,
        "correction_projection_coefficients": coefficients,
        "weighted_residual_rms_pu": float(np.sqrt(weights @ residual**2)),
        "weighted_input_residual_rms_pu": float(np.sqrt(weights @ b**2)),
        "weighted_projection_rms_pu": float(np.sqrt(weights @ projection**2)),
        "weighted_projection_orthogonal_inner_product": float(weights @ (projection * residual)),
        "weights_sum": float(weights.sum()), "event_weight_sums": [float(weights[sum(len(q[0]) for q in valid[:i]):sum(len(q[0]) for q in valid[:i+1])].sum()) for i in range(event_count)]}


def svd_integration_sensitivity(simpson, trapezoid, fit):
    """No parameter refit: compare both designs with the primary column scaling."""
    if fit.get("rank") != 2:
        return {"eta": None, "condition_gate": False, "eta_gate": False}
    count = len(simpson)
    weights = np.concatenate([np.full(len(x), 1/(count*len(x))) for x in simpson])
    delta = (np.concatenate(simpson)-np.concatenate(trapezoid))*np.sqrt(weights[:,None]) / fit["column_weighted_rms"]
    # Matrix 2-norm from the two-by-two Gram eigenvalues; not another fit/SVD.
    norm = float(np.sqrt(max(0., np.linalg.eigvalsh(delta.T @ delta)[-1])))
    eta = norm / fit["singular_values_scaled"][-1]
    return {"eta": eta, "condition_gate": fit["condition_scaled"] <= 1000, "eta_gate": eta <= .1}


def rate_residual_stats(xs, xt, ys, beta):
    """Event/within-event-window equal normalized rate residuals, fixed beta."""
    if not xs or any(not len(x) for x in xs):
        return {"Q_S": None, "Q_T": None, "nu": None, "status": "unknown"}
    qs = [y-x@beta for x,y in zip(xs,ys)]
    qt = [y-x@beta for x,y in zip(xt,ys)]
    rms = lambda arrays: float(np.sqrt(np.mean([np.mean(z*z) for z in arrays])))
    return {"Q_S": rms(qs), "Q_T": rms(qt), "nu": max(1e-10, rms([s-t for s,t in zip(qs,qt)])),
            "per_event_q_S": qs, "per_event_q_T": qt, "status": "computed"}


def artificial_qualification():
    """Exactly the finite scalar fixture registered by Scientist M1."""
    from scipy.linalg import expm
    max_error = 0.
    cases = 0
    for a in (0., 1e-10, 1., 100.):
        for slope in (0., .4):
            for h in (.005, .01):
                g, p0, w0 = .7, .2, -.3
                matrix = np.array([[-a,-g,0],[0,0,1],[0,0,0]], dtype=np.float64)
                expected = (expm(h*matrix) @ [p0,w0,slope])[0]
                omega = np.array([[w0],[w0+slope*h]])
                forward = linear_filter(omega,[p0],[g],[a],h)
                reverse = linear_filter(omega[::-1],forward[-1],[g],[a],-h)
                np.testing.assert_allclose(forward[-1,0],expected,atol=1e-12,rtol=1e-12)
                np.testing.assert_allclose(reverse[-1,0],p0,atol=1e-12,rtol=1e-12)
                if a == 0:
                    np.testing.assert_allclose(forward[-1,0],p0-g*h*(omega[0,0]+omega[1,0])/2,atol=1e-12,rtol=1e-12)
                if slope == 0 and a:
                    np.testing.assert_allclose(forward[-1,0],np.exp(-a*h)*p0-g*w0*h*phi12(-a*h)[0],atol=1e-12,rtol=1e-12)
                max_error = max(max_error,abs(forward[-1,0]-expected),abs(reverse[-1,0]-p0))
                cases += 1
    return {"status": "PASS", "cases": cases, "atol": 1e-12, "rtol": 1e-12, "max_abs_error": max_error, "new_FOM": 0}


def fixed_coefficient_residual(event_designs, event_residuals, coefficients):
    values = [np.mean((np.asarray(b) - np.asarray(x) @ coefficients)**2) for x, b in zip(event_designs, event_residuals) if len(x)]
    return float(np.sqrt(np.mean(values))) if values else None


def scope_steps(event, scope):
    if event["label"] == "stable" and scope in ("stable5", "stable8"):
        n = 500 if scope == "stable5" else 800
        return n if len(event["x"]) >= n + 11 else None
    if event["label"] == "unstable" and scope == "unstable_actual15":
        return min(1500, len(event["x"]) - 11)
    return None


def effective_parameters(model, export):
    """Read the actual loaded buffers and require exact exported-state agreement."""
    values = {}
    checks = {}
    for name in ("jpi_p", "rpp", "storage_base", "mass", "p_scale"):
        value = getattr(model, name).detach().cpu().numpy().astype(np.float64, copy=True)
        if value.shape != (10,) or not np.isfinite(value).all():
            raise ValueError("invalid effective mechanical buffer")
        values[name] = value
        checks[name] = bool(np.array_equal(value, np.asarray(export["state_dict"][name])))
    if not all(checks.values()):
        raise ValueError("loaded model differs from effective state_dict")
    values["g"] = values["jpi_p"]
    values["a"] = values["rpp"] * values["storage_base"]
    backward = model.p_backward.detach().cpu().numpy()
    from scipy.linalg import expm
    matrix = np.zeros((30,30))
    matrix[:10,:10], matrix[:10,10:20], matrix[10:20,20:] = -np.diag(values["a"]), -np.diag(values["g"]), np.eye(10)
    backward_expected = expm(-.01*matrix)[:10]
    np.testing.assert_allclose(backward_expected, reverse_matrix(values["g"],values["a"]), atol=1e-12,rtol=1e-12)
    return values, {"effective_buffer_export_identity": checks,
        "p_backward_loaded_vs_rebuilt_maxabs": float(np.max(np.abs(backward-backward_expected))),
        "p_backward_loaded_vs_rebuilt_allclose_1e12": bool(np.allclose(backward, backward_expected, atol=1e-12, rtol=1e-12)),
        "values": values, "equation": "p_dot=-jpi_p*omega-rpp*storage_base*p; momentum port=+jpi_p*storage_base*p",
        "units": {"time": "s", "omega": "rad/s synchronous deviation; (raw omega-1)*omega_s", "p": "Pm_raw-base.pm_star; system-base pu is the source adapter declaration", "g": "declared pu/rad", "a": "1/s", "p_scale": "normalization only, absent from p dynamics", "storage_energy_SI": "unknown"},
        "coordinate_definition_traceable": True, "base_pm_star_equals_FOM_equilibrium_Pm": "unknown",
        "physical_equilibrium_origin_attested": False,
        "interpretation_scope": "conditional diagnosis in frozen cache coordinates; unresolved power-base/origin cannot establish physical first-order-form inadequacy",
        "node_order": "shared ten-column frozen model/cache order; named generator-to-column mapping unknown unless attested by frozen source",
        "source_chain": ["force_graph_model_20261028.strict_m3_config: softplus(raw structured field)*mask", "ForceGraphROM buffers and exported state_dict", "StateGatedMemoryROM.field and MemoryTransportROM inherited p block"],
        "unit_source": ["src/kiki/data/private_dataset.py:canonical_arrays", "train_local_memory_0098.read_train: Pm_raw-base.pm_star, not can.Pm_star", "experiments/closure_discovery_data.py: canonical x=(delta-eq,(omega-1)*OMEGA_S,pm-base_model.pm_star)"],
        "automatic_rescaling": False}


def source_metadata_provenance(events, model):
    """Root-only HDF5 metadata via the existing explicitly events-free helper."""
    result = {"asset_identity": "verified_frozen_model_cache", "coordinate_alignment": "unknown", "equilibrium_reference_alignment": "unknown", "physical_unit_provenance": "unknown"}
    try:
        import closure_discovery_data as data
        from train_local_memory_0098 import BASE
        reference = data._load_reference_fingerprint()
        payload = data._reference_fingerprint_payload(reference)
        with np.load(BASE,allow_pickle=False) as base:
            base_pm = base["pm_star"].copy()
            base_mass = base["mass"].copy()
        eq_pm = np.asarray(reference["equilibrium"]["Pm"])
        difference = base_pm-eq_pm
        mapping = reference.get("state_mapping")
        mass = model.mass.detach().cpu().numpy()
        node_mass_ok = np.allclose(base_mass,reference["inertia"]/reference["omega_s"],atol=1e-12,rtol=1e-12) and np.allclose(mass,base_mass,atol=1e-12,rtol=1e-12)
        source_matches = []
        for event in events:
            root = str(event.get("root", ""))
            if root in reference["sources"]:
                source_matches.append(True)
            else:
                meta = event.get("classification_meta",{})
                source_matches.append(meta.get("system_fingerprint_sha256") == reference["fingerprint_sha256"] and meta.get("system_attestation_complete") is True)
        result.update(base=ident(BASE), base_pm_star=base_pm, FOM_equilibrium_Pm=eq_pm,
            equilibrium_difference= difference, equilibrium_difference_maxabs=float(np.max(np.abs(difference))),
            equilibrium_reference_alignment="matched_at_1e12" if np.allclose(base_pm,eq_pm,atol=1e-12,rtol=1e-12) else "mismatched",
            coordinate_alignment="matched_source_mapping_derivation_and_fingerprint" if bool(mapping) and all(source_matches) else "unknown",
            coordinate_order_evidence="fixed ten-column reader schema; original state_mapping and inherited unpermuted model buffers; frozen cache/initial-state identity; fresh source fingerprint",
            effective_mass_equals_base_and_physical_inertia=bool(node_mass_ok), mass_equality_used_as_order_gate=False,
            model_mass=mass, base_mass=base_mass, FOM_inertia=reference["inertia"],
            fit_source_metadata_covered=sum(source_matches), fit_source_metadata_denominator=len(events),
            physical_unit_provenance="documented_frozen_metadata; no independent ANDES internal conversion attestation", raw_units=reference["raw_units"], mva_base=reference["mva_base"], omega_s=reference["omega_s"],
            historical_metadata_fingerprint=reference["fingerprint_sha256"], metadata_fragment=payload,
            metadata_fragment_sha256=hashlib.sha256(json.dumps(safe(payload),sort_keys=True,allow_nan=False).encode()).hexdigest(),
            original_HDF5_whole_file_rehashed=False, events_group_opened=False,
            source_HDF5_files=[f"private_data/{name}/dataset.h5:root attrs/equilibrium/inertia only" for name in reference["sources"]])
    except Exception as exc:
        result["unknown_reason"] = repr(exc)
    return result


def load_sources(protocol, design):
    """Bind previous complete receipts; hash each prediction once when consumed."""
    import checkpoint_selection_20261030 as old
    previous_manifest_path = PREVIOUS / "manifest.json"
    previous = read(previous_manifest_path)
    previous_hash = sha(previous_manifest_path)
    if previous_hash != "9413d62bab914500c2049506f01fd8a70998ac1c7f011c06f8574242a4bc0686":
        raise ValueError("Scientist-frozen previous manifest hash differs")
    models, summaries, records, bindings = {}, {}, {}, {}
    import closure_probe_20261030 as probe
    for label in MODELS:
        folder = PREVIOUS / label
        receipt = read(folder / "complete.json")
        summary = read(folder / "summary.json")
        if label == "step000016" and sha(folder / "summary.json") != "810e9caa0ad1e4f86d31f6ac481e7d71c50562f54e3425beb4db824c209d0d77":
            raise ValueError("Scientist-frozen A16 summary differs")
        if receipt.get("status") != "COMPLETED" or receipt["manifest_sha256"] != previous_hash or receipt["summary_sha256"] != sha(folder / "summary.json") or summary["manifest_sha256"] != previous_hash or summary["checkpoint"] != label:
            raise ValueError("previous completion binding differs")
        if summary["model"] != {key: previous["models"][label][key] for key in ("path", "sha256")} or sha(ROOT / summary["model"]["path"]) != summary["model"]["sha256"]:
            raise ValueError("frozen model identity changed")
        detail_ident = summary["per_event"]
        if detail_ident not in receipt["artifacts"] or sha(ROOT / detail_ident["path"]) != detail_ident["sha256"]:
            raise ValueError("previous per-event receipt differs")
        detail = read(ROOT / detail_ident["path"])
        if detail["timing_actual15"] != summary["decision_profile"]["timing_actual15"]:
            raise ValueError("qualification timing profile changed")
        for pred in summary["prediction_files"]:
            item = {"path": Path(pred["path"]).resolve().relative_to(ROOT.resolve()).as_posix(), "sha256": pred["sha256"]}
            if item not in receipt["artifacts"]:
                raise ValueError("prediction not bound by previous receipt")
            allowed = (ROOT / "private_data/state_gated_memory_20261028/study01/study/A_28311/fit000096") if label == "A96" else PREVIOUS / "step000016/attempt001/paired"
            Path(pred["path"]).resolve().relative_to(allowed.resolve())
        # Non-prediction artifacts are small: verify the entire original receipt;
        # required prediction bytes are hashed once while consumed below.
        prediction_paths = {Path(e["path"]).resolve() for e in summary["prediction_files"]}
        for artifact in receipt["artifacts"]:
            path = (ROOT / artifact["path"]).resolve()
            path.relative_to(ROOT.resolve())
            if path not in prediction_paths and sha(path) != artifact["sha256"]:
                raise ValueError("prior completion artifact changed")
        summaries[label], records[label] = summary, detail
        spec = old.model_spec(label)
        models[label] = probe.load_model(spec)
        bindings[label] = {"summary": ident(folder / "summary.json"), "complete": ident(folder / "complete.json"), "per_event": detail_ident, "model": summary["model"], "predictions": summary["prediction_files"]}
    cache = ROOT / previous["data"]["fit_cache"]["path"]
    if sha(cache) != previous["data"]["fit_cache"]["sha256"]:
        raise ValueError("frozen fit cache changed")
    # Existing cache format; load only fit payloads, never no_update/val/test.
    with np.load(cache, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        events = [{**event, "x": data[f"event_{i:04d}_x"].copy()} for i, event in enumerate(metadata)]
    labels = {event["id"]: event["label"] for event in events}
    if len(events) != 297 or labels != previous["data"]["event_labels"] or list(labels.values()).count("stable") != 267 or list(labels.values()).count("unstable") != 30:
        raise ValueError("fit297 identity or 267/30 denominator changed")
    if any(np.asarray(e["x"]).dtype != np.float64 or e["x"].shape[1] != 30 for e in events):
        raise ValueError("frozen coordinate array changed")
    code_names = ("p_port_diagnostic_20261031.py", "checkpoint_selection_20261030.py", "closure_probe_20261030.py", "state_gated_memory_model_20261028.py", "memory_transport_model_20261029.py", "force_graph_model_20261028.py", "closure_discovery_data.py", "train_local_memory_0098.py")
    manifest = {"schema": SCHEMA, "protocol": ident(protocol), "design": ident(design), "previous_manifest": ident(previous_manifest_path),
        "models": bindings, "data": previous["data"], "source": {name: ident(ROOT / "experiments" / name) for name in code_names},
        "canonical_adapter": ident(ROOT / "src/kiki/data/private_dataset.py"), "scopes": list(SCOPES),
        "optimizer_calls": 0, "new_ROM_rollouts": 0, "val_read": False, "test_read": False, "new_fom": 0}
    manifest["M0_source_metadata"] = source_metadata_provenance(events, models["A96"])
    write_once(PRIVATE / "manifest.json", manifest)
    return models, summaries, records, events, manifest


def summarize_decompositions(rows):
    result = {}
    for scope in SCOPES:
        selected = [r for r in rows if r["scope"] == scope]
        expected = 30 if scope == "unstable_actual15" else 267
        ok = len(selected) == expected and all(r["qualified"] and r.get("fine") is not None for r in selected)
        cell = {"expected_events": expected, "recorded_events": len(selected), "qualified_events": sum(r["qualified"] for r in selected), "all_required_rows_qualified": ok, "event_equal_aggregation": True}
        if ok:
            for kind in ("per_machine_mse_pu2", "per_machine_normalized_mse"):
                cell[kind] = {term: np.mean([r["fine"][kind][term] for r in selected], axis=0) for term in selected[0]["fine"][kind]}
            cell["rmse_pu"] = {term: float(np.sqrt(np.mean(cell["per_machine_mse_pu2"][term]))) for term in ("implementation", "omega_transmission", "fixed_truth_driven", "total")}
            cell["normalized_mse"] = {term: float(np.mean(value)) for term, value in cell["per_machine_normalized_mse"].items()}
            tau = float(np.sqrt(np.mean([r["tau_e"]**2 for r in selected])))
            fixed = math.sqrt(cell["normalized_mse"]["fixed_truth_driven"])
            omega = math.sqrt(cell["normalized_mse"]["omega_transmission"])
            cell.update(tau_G=tau, R_fixed=fixed, R_omega=omega, fixed_residual_resolved=fixed>10*tau,
                fixed_residual_resolved_events=sum(r["fixed_residual_resolved"] for r in selected),
                self_closure_passed_events=sum(r["self_closure_passed"] for r in selected),
                all_self_closure_passed=all(r["self_closure_passed"] for r in selected),
                operational_pattern="mixed" if fixed>10*tau and omega>10*tau else "omega_dominant_at_current_scale" if fixed<=10*tau and omega>10*tau else "fixed_residual_resolved" if fixed>10*tau else "unresolved_at_current_scale")
        else:
            cell.update(rmse_pu=None, normalized_mse=None)
        result[scope] = cell
    return result


def shared_constant_analysis(events, parameters, m2_rows, provenance):
    """Exactly ten absolute-beta fits on stable8 + unstable actual, no intercept."""
    arrays, machines = {}, []
    if provenance["coordinate_alignment"]=="unknown":
        return {"status":"unknown","reason":"coordinate_or_node_alignment_unknown","expected_parameter_sets":10,"estimated_parameter_sets":0,"machines":[],"structure_scope_summary":{}},arrays
    pscale = parameters["A96"]["p_scale"]
    for machine in range(10):
        collection = {}
        for scope in SCOPES:
            items, xs, xt, ys, excluded, tails = [], [], [], [], [], {}
            for event in events:
                n = scope_steps(event,scope)
                if n is None:
                    continue
                truth = event["x"][10:n+11, [10+machine,20+machine]]
                Is, delta, tail = window_integrals(truth,"simpson")
                It, _, _ = window_integrals(truth,"trapezoid")
                if not len(Is) or not np.isfinite(truth).all():
                    excluded.append(event["id"])
                items.append(event["id"])
                xs.append(Is/(.02*pscale[machine]))
                xt.append(It/(.02*pscale[machine]))
                ys.append(-delta[:,1]/(.02*pscale[machine]))
                tails[event["id"]] = tail
            collection[scope] = (items,xs,xt,ys,excluded,tails)
        joined = [collection["stable8"],collection["unstable_actual15"]]
        xs,xt,ys = ([value for group in joined for value in group[index]] for index in (1,2,3))
        exclusions = [eid for group in joined for eid in group[4]]
        if exclusions or len(xs)!=297:
            machines.append({"machine_column":machine,"status":"unknown","reason":"incomplete_original_297_windows_no_reweighting","unavailable_event_ids":exclusions})
            continue
        fit = event_equal_svd(xs,ys)
        beta = fit.pop("correction_projection_coefficients")
        fit["absolute_g_a"] = beta
        sensitivity = svd_integration_sensitivity(xs,xt,fit)
        identifiable = fit["rank"] == 2 and sensitivity["condition_gate"] and sensitivity["eta_gate"]
        signs = bool(beta[0]>0 and beta[1]>=0)
        scope_reports = {}
        for scope,(ids,xs_s,xt_s,ys_s,missing,tails) in collection.items():
            fitted = rate_residual_stats(xs_s,xt_s,ys_s,beta)
            reports, baseline_cache = {}, {}
            for label in MODELS:
                beta0 = np.array([parameters[label]["g"][machine],parameters[label]["a"][machine]])
                key=tuple(beta0)
                if key not in baseline_cache:
                    baseline_cache[key]=rate_residual_stats(xs_s,xt_s,ys_s,beta0)
                baseline=baseline_cache[key]
                for name, cell in (("baseline",baseline),("estimated",fitted)):
                    for scheme in ("S","T"):
                        key=f"m{machine}_{scope}_{label}_{name}_q{scheme}"
                        values=cell.get(f"per_event_q_{scheme}",[])
                        arrays[key]=np.concatenate(values) if values else np.empty(0)
                numeric = not missing and baseline["Q_S"] is not None and fitted["Q_S"] is not None
                resolved0 = numeric and min(baseline["Q_S"],baseline["Q_T"])>10*baseline["nu"]
                explained = numeric and max(fitted["Q_S"],fitted["Q_T"])<=10*fitted["nu"] and signs
                remaining = numeric and min(fitted["Q_S"],fitted["Q_T"])>10*fitted["nu"] and min(fitted["Q_S"],fitted["Q_T"])>=.2*max(baseline["Q_S"],baseline["Q_T"])
                source = provenance["coordinate_alignment"] != "unknown" and provenance["equilibrium_reference_alignment"] == "matched_at_1e12" and provenance.get("mechanical_reverse_blocks_qualified",False)
                selected=[r for r in m2_rows[label] if r["scope"]==scope]
                original_and_m1 = len(selected)==(30 if scope=="unstable_actual15" else 267) and all(r["qualified"] and r.get("self_closure_passed",False) for r in selected)
                fixed_resolved = len(selected)>0 and all(r.get("tau_e") is not None for r in selected) and np.sqrt(np.mean([r["fixed_error_normalized_rms"]**2 for r in selected]))>10*np.sqrt(np.mean([r["tau_e"]**2 for r in selected]))
                prerequisites = source and identifiable and original_and_m1 and fixed_resolved
                clean=lambda cell:{k:v for k,v in cell.items() if not k.startswith("per_event_q_")}
                reports[label]={"baseline":clean(baseline),"estimated":clean(fitted),"baseline_resolved":bool(resolved0),"shared_constants_explain_at_current_scale":bool(resolved0 and explained),"remaining_form_pattern":bool(remaining),
                    "interpretation_prerequisites_passed":bool(prerequisites),"source_equilibrium_and_coordinates":bool(source),"all_original_and_self_closure_qualified":bool(original_and_m1),"M2_fixed_residual_resolved":bool(fixed_resolved),
                    "interpretation":"shared_constant_compatible" if prerequisites and resolved0 and explained else "remaining_form_evidence_in_frozen_coordinates" if prerequisites and remaining else "unknown"}
            scope_reports[scope]={"original_event_denominator":30 if scope=="unstable_actual15" else 267,"event_ids":ids,"window_counts":[len(x) for x in xs_s],"dropped_10ms_tail_by_event":tails,"unavailable_event_ids":missing,"models":reports}
        machines.append({"machine_column":machine,"status":"computed","fit":fit,"integration_sensitivity":sensitivity,"independent_constant_identifiability":bool(identifiable),"signs_legal_g_positive_a_nonnegative":signs,"scopes":scope_reports})
    structure = {}
    for label in MODELS:
        supporting=[]
        for machine in machines:
            if machine["status"]!="computed":continue
            yes=lambda s: machine["scopes"][s]["models"][label]["interpretation"]=="remaining_form_evidence_in_frozen_coordinates"
            if (yes("stable5") and yes("stable8")) or yes("unstable_actual15"):
                supporting.append(machine["machine_column"])
        structure[label]={"machine_columns_meeting_frozen_scope_rule":supporting,"automatic_structure_adoption":False}
    return {"expected_parameter_sets":10,"estimated_parameter_sets":sum(m["status"]=="computed" for m in machines),"fit_scope":"stable complete8 plus unstable actual<=15 once; stable5 report only","normalization":"A96 frozen p_scale; normalized p-change rates per second","machines":machines,"structure_scope_summary":structure}, arrays


def run(protocol, design):
    if PRIVATE.exists() or PUBLIC.exists():
        raise FileExistsError("immutable diagnostic already exists; reconcile prior job, never automatically replay")
    PRIVATE.mkdir(parents=True)
    started = time.perf_counter()
    write_once(PRIVATE / "started.json", {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "optimizer_calls": 0, "new_ROM_rollouts": 0})
    try:
        stage=time.perf_counter()
        models, summaries, qualifications, events, manifest = load_sources(protocol, design)
        read_seconds=time.perf_counter()-stage
        artificial=artificial_qualification()
        parameters, m0 = {}, {}
        for label in MODELS:
            parameters[label], m0[label] = effective_parameters(models[label], read(ROOT / summaries[label]["model"]["path"]))
            m0[label]["source_metadata"]=manifest["M0_source_metadata"]
            m0[label]["base_pm_star_equals_FOM_equilibrium_Pm"]=manifest["M0_source_metadata"]["equilibrium_reference_alignment"]
            m0[label]["physical_equilibrium_origin_attested"]=manifest["M0_source_metadata"]["equilibrium_reference_alignment"]=="matched_at_1e12"
        # Derived from the loaded buffers; does not change the registered manifest.
        provenance={**manifest["M0_source_metadata"],"mechanical_reverse_blocks_qualified":all(m0[label]["p_backward_loaded_vs_rebuilt_allclose_1e12"] for label in MODELS)}
        same_constants = all(np.array_equal(parameters[MODELS[0]][key], parameters[MODELS[1]][key]) for key in ("g", "a"))
        coordinate_allowed=manifest["M0_source_metadata"]["coordinate_alignment"]!="unknown"
        truth_cache, rows, histories = {}, {}, {}
        lookup = {e["id"]: e for e in events}
        stage=time.perf_counter()
        for label in MODELS:
            print(json.dumps({"model": label, "status": "STARTED", "new_ROM_rollouts": 0}), flush=True)
            p = parameters[label]
            allrows, historyrows, seen = [], [], []
            q = {r["event_id"]: r for r in qualifications[label]["classification"]["8s"]["rows"]}
            qa = {r["event_id"]: r for r in qualifications[label]["classification"]["actual15"]["rows"]}
            for entry in summaries[label]["prediction_files"]:
                path = Path(entry["path"])
                if sha(path) != entry["sha256"]:
                    raise ValueError("frozen prediction hash changed")
                with np.load(path, allow_pickle=False) as block:
                    ids = [str(x) for x in block["event_ids"]]
                    if block["fine"].shape != (len(ids), 3001, 40) or block["coarse"].shape != (len(ids), 1501, 40) or block["histories"].shape != (len(ids), 11, 30):
                        raise ValueError("paired array shape changed")
                    for eid, h, fine, coarse in zip(ids, block["histories"], block["fine"], block["coarse"]):
                        event = lookup[eid]
                        if not np.array_equal(h, event["x"][:11]) or not np.allclose(fine[0, :30], h[-1],atol=1e-12,rtol=1e-12) or not np.allclose(coarse[0, :30], h[-1],atol=1e-12,rtol=1e-12):
                            raise ValueError("prediction event/history/initial observed state mismatch")
                        seen.append(eid)
                        backward = linear_filter(h[::-1, 10:20], h[-1, 20:30], p["g"], p["a"], -.01)[::-1]
                        backward_right=right_endpoint_filter(h[::-1,10:20],h[-1,20:30],p["g"],p["a"],-.01)[::-1]
                        err = backward[:-1] - h[:-1, 20:30] if coordinate_allowed else None
                        original_backward = models[label].p_backward.detach().cpu().numpy()
                        reconstructed = [h[-1, 20:30]]
                        for k in range(9, -1, -1):
                            slope = (h[k+1, 10:20] - h[k, 10:20]) / .01
                            reconstructed.append(original_backward @ np.r_[reconstructed[-1], h[k+1, 10:20], slope])
                        history_r=normalized_rms(err,p["p_scale"]) if coordinate_allowed else None
                        history_s=normalized_rms(backward[:-1]-backward_right[:-1],p["p_scale"])
                        original_difference=float(np.max(np.abs(backward-np.asarray(reconstructed[::-1]))))
                        reverse_ok=bool(np.allclose(backward,np.asarray(reconstructed[::-1]),atol=1e-12,rtol=1e-12))
                        historyrows.append({"event_id": eid, "label": event["label"], "status":"computed" if coordinate_allowed else "unknown", "per_machine_rmse_pu": np.sqrt(np.mean(err**2, axis=0)) if err is not None else None, "per_machine_normalized_mse": np.mean((err / p["p_scale"])**2, axis=0) if err is not None else None, "original_p_backward_maxabs_difference": original_difference,
                            "original_reverse_block_qualified":reverse_ok,"history_normalized_rms":history_r,"right_endpoint_sensitivity":history_s,"history_incompatible_at_scale":bool(history_r>10*max(1e-10,history_s)) if history_r is not None else None})
                        for scope in SCOPES:
                            n = scope_steps(event, scope)
                            if n is None:
                                continue
                            qualified = (qa if scope == "unstable_actual15" else q)[eid]["numerically_admissible"]
                            record = {"event_id": eid, "label": event["label"], "scope": scope, "future_steps": n, "qualified": qualified,"original_qualified":qualified}
                            ff, cc, truth = fine[:2*n+1], coarse[:n+1], event["x"][10:n+11]
                            if not all(np.isfinite(v).all() for v in (ff, cc, truth)):
                                record.update(qualified=False, fine=None, coarse=None, reason="nonfinite_cached_interval")
                                allrows.append(record)
                                continue
                            fm = linear_filter(ff[:, 10:20], h[-1, 20:30], p["g"], p["a"], .005)[::2]
                            cm = linear_filter(cc[:, 10:20], h[-1, 20:30], p["g"], p["a"], .01)
                            dm=linear_filter(ff[::2,10:20],h[-1,20:30],p["g"],p["a"],.01)
                            sensitivities={"predicted_p_fine_coarse":normalized_rms(ff[::2][1:,20:30]-cc[1:,20:30],p["p_scale"]),"filter_fine_coarse":normalized_rms(fm[1:]-cm[1:],p["p_scale"]),"fine_input_interpolation":normalized_rms(fm[1:]-dm[1:],p["p_scale"])}
                            s=max(sensitivities.values())
                            r=normalized_rms(ff[::2][1:,20:30]-fm[1:],p["p_scale"])
                            record.update(sensitivity_components=sensitivities,s_e=s,r_e=r,self_closure_passed=self_closure_gate(r,s))
                            if not coordinate_allowed:
                                record.update(fine=None,coarse=None,qualified=False,reason="M0_coordinate_or_node_alignment_unknown")
                                allrows.append(record)
                                continue
                            key = (eid, "shared" if same_constants else label)
                            if key not in truth_cache:
                                maxn=800 if event["label"]=="stable" else min(1500,len(event["x"])-11)
                                full_truth=event["x"][10:maxn+11]
                                truth_cache[key] = (linear_filter(full_truth[:, 10:20], h[-1, 20:30], p["g"], p["a"], .01),right_endpoint_filter(full_truth[:,10:20],h[-1,20:30],p["g"],p["a"],.01))
                            tf,tr = (value[:n+1] for value in truth_cache[key])
                            record["fine"] = squared_decomposition(ff[::2][1:, 20:30], fm[1:], tf[1:], truth[1:, 20:30], p["p_scale"])
                            record["coarse"] = squared_decomposition(cc[1:, 20:30], cm[1:], tf[1:], truth[1:, 20:30], p["p_scale"])
                            right_s=normalized_rms(tf[1:]-tr[1:],p["p_scale"])
                            tau=max(1e-10,r,s,right_s)
                            fixed_error=normalized_rms(tf[1:]-truth[1:,20:30],p["p_scale"])
                            record.update(sensitivity_components=sensitivities,s_e=s,r_e=r,tau_e=tau,right_endpoint_sensitivity=right_s,
                                self_closure_passed=self_closure_gate(r,s),fixed_error_normalized_rms=fixed_error,fixed_residual_resolved=bool(fixed_error>10*tau))
                            allrows.append(record)
            if len(seen) != 297 or len(set(seen)) != 297 or set(seen) != set(lookup):
                raise ValueError("prediction ID cohort incomplete or duplicated")
            rows[label], histories[label] = allrows, historyrows
            print(json.dumps({"model": label, "status": "COMPLETED", "events": len(seen)}), flush=True)
        filter_seconds=time.perf_counter()-stage
        stage=time.perf_counter()
        m3,m3_arrays=shared_constant_analysis(events,parameters,rows,provenance)
        estimate_seconds=time.perf_counter()-stage
        aggregate = {label: summarize_decompositions(rows[label]) for label in MODELS}
        paired = {}
        for scope in SCOPES:
            left = {r["event_id"]: r for r in rows[MODELS[0]] if r["scope"] == scope}
            right = {r["event_id"]: r for r in rows[MODELS[1]] if r["scope"] == scope}
            paired[scope] = {eid: {term: np.mean(right[eid]["fine"]["per_machine_normalized_mse"][term]) - np.mean(left[eid]["fine"]["per_machine_normalized_mse"][term]) for term in left[eid]["fine"]["per_machine_normalized_mse"]} if left[eid]["fine"] is not None and right[eid]["fine"] is not None else None for eid in sorted(left)}
        result = {"schema": SCHEMA, "status": "COMPLETED", "manifest_sha256": sha(PRIVATE / "manifest.json"), "M0": m0, "M1_artificial_qualification":artificial,"same_g_a": same_constants,"same_p_scale":bool(np.array_equal(parameters[MODELS[0]]["p_scale"],parameters[MODELS[1]]["p_scale"])), "M2_aggregates": aggregate,
            "M3": m3,
            "M4": {label: {"events": len(histories[label]),"coordinate_comparison_status":"computed" if coordinate_allowed else "unknown", "per_machine_rmse_pu_event_equal": np.sqrt(np.mean([r["per_machine_rmse_pu"]**2 for r in histories[label]], axis=0)) if coordinate_allowed else None, "original_reverse_block_maxabs_difference": max(r["original_p_backward_maxabs_difference"] for r in histories[label]),"history_incompatible_event_count":sum(r["history_incompatible_at_scale"] for r in histories[label]) if coordinate_allowed else None,"all_reverse_blocks_qualified":all(r["original_reverse_block_qualified"] for r in histories[label])} for label in MODELS},
            "runtime": {"read_and_bind_seconds":read_seconds,"filter_seconds":filter_seconds,"shared_estimation_seconds":estimate_seconds,"wall_seconds": time.perf_counter() - started, "optimizer_calls": 0, "new_ROM_rollouts": 0, "new_fom": 0},
            "interpretation": "offline consumed-train diagnostic, not deployed prediction improvement or model adoption", "model_adopted": False}
        write_once(PRIVATE / "per_event.json", {"M2": rows, "M4": histories, "paired_normalized_mse_delta_step16_minus_A96": paired})
        with (PRIVATE / "M3_rate_residuals.npz").open("xb") as handle:
            np.savez_compressed(handle,**m3_arrays)
        write_once(PRIVATE / "diagnostic.json", result)
        public = {key: value for key, value in result.items() if key not in ("M3","M0")}
        public["M0"]={label:{"effective_buffer_export_identity":m0[label]["effective_buffer_export_identity"],"p_backward_loaded_vs_rebuilt_allclose_1e12":m0[label]["p_backward_loaded_vs_rebuilt_allclose_1e12"],"p_backward_loaded_vs_rebuilt_maxabs":m0[label]["p_backward_loaded_vs_rebuilt_maxabs"],"source_metadata":{k:v for k,v in manifest["M0_source_metadata"].items() if k in ("asset_identity","coordinate_alignment","equilibrium_reference_alignment","physical_unit_provenance","equilibrium_difference_maxabs","node_mass_order_check","fit_source_metadata_covered","fit_source_metadata_denominator","metadata_fragment_sha256","events_group_opened")}} for label in MODELS}
        public["M3"]={key:value for key,value in m3.items() if key!="machines"}
        public["M3"]["machines"]=[]
        for machine in m3["machines"]:
            copy={key:value for key,value in machine.items() if key not in ("scopes","unavailable_event_ids")}
            if "scopes" in machine:
                copy["scopes"]={scope:{key:value for key,value in cell.items() if key not in ("event_ids","window_counts","dropped_10ms_tail_by_event","unavailable_event_ids")} | {"unavailable_event_count":len(cell["unavailable_event_ids"]),"dropped_tail_event_count":sum(value>0 for value in cell["dropped_10ms_tail_by_event"].values())} for scope,cell in machine["scopes"].items()}
            public["M3"]["machines"].append(copy)
        write_once(PUBLIC, public)
        write_once(PRIVATE / "complete.json", {"status": "COMPLETED", "diagnostic": ident(PRIVATE / "diagnostic.json"), "per_event": ident(PRIVATE / "per_event.json"),"M3_rate_residuals":ident(PRIVATE/"M3_rate_residuals.npz"), "public": ident(PUBLIC), "manifest_sha256": result["manifest_sha256"]})
        return result
    except BaseException as exc:
        write_once(PRIVATE / "failure.json", {"status": "FAILED", "error": repr(exc), "traceback": traceback.format_exc(), "wall_seconds": time.perf_counter() - started, "optimizer_calls": 0, "new_ROM_rollouts": 0, "failure_is_physical_instability": False})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run"])
    parser.add_argument("--protocol", type=Path, default=ROOT / "reports/p_port_diagnostic_20261031_protocol.md")
    parser.add_argument("--design", type=Path, default=ROOT / "reports/p_port_diagnostic_20261031_scientist.md")
    args = parser.parse_args(argv)
    run(args.protocol, args.design)


if __name__ == "__main__":
    main()
