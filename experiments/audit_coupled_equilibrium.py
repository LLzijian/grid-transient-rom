"""Loop0065: fixed coupled-equilibrium and storage audit (development-free)."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import solve_continuous_lyapunov

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from kiki.models.private_pm import MechanicalROM
from audit_long_latent_rollout import BASE, MODEL, params
from evaluate_long_latent import NETWORK, _construct

MATRICES = ROOT / "private_data/task2_long_sparse_audit_0061/matrices.npz"
VERIFIED = ROOT / "private_data/task2_verified_modes_0035b"
EXPECTED_MODEL = "d34d9f5c36fb296e1d49e59f5f7b18a5da33b56bf371a16169f82bc25664995c"
EXPECTED_VERIFIED = "8a1d3e53a1b4aeba5ceb2f1c7017378dc3ceb51be64a372a3562da8b9a7970ef"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b) / np.maximum(1., np.abs(a))))


def _jsonable(value):
    """Evidence must fail loudly rather than serialize a nonfinite value."""
    if isinstance(value, dict): return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray): return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value): raise FloatingPointError("nonfinite evidence")
        return float(value)
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.bool_): return bool(value)
    return value


def analytic_jacobian(p: dict[str, np.ndarray]) -> np.ndarray:
    """Exact 33-state linearization; supports small synthetic angle fixtures."""
    h = np.asarray(p["angle_incidence"], float)
    nedge = h.shape[1]
    xi = np.asarray(p["xi"], float) * np.asarray(p["support"], float)
    alpha = np.asarray(p["alpha"], float) * np.asarray(p["alpha_support"], float)
    scale = np.asarray(p["feature_scale"], float)
    if xi.shape[0] != 24 + 2 * nedge or scale.size != 20 + 2 * nedge:
        raise ValueError("feature/angle dimensions")
    phase = np.asarray(p["delta_star"], float) @ h
    gd = np.vstack((np.diag(np.cos(phase) / scale[20:20+nedge]) @ h.T,
                    np.diag(-np.sin(phase) / scale[20+nedge:20+2*nedge]) @ h.T))
    ad = alpha.T @ gd
    bd = xi[24:24+2*nedge].T @ gd
    bw = xi[4:14].T @ np.diag(1 / scale[:10])
    bp = xi[14:24].T @ np.diag(1 / scale[10:20])
    aq = xi[1:4].T
    inc = np.asarray(p["incidence"], float); mass = np.asarray(p["mass"], float)
    stiffness = (inc * (np.asarray(p["coupling"], float) * np.cos(np.asarray(p["delta_star"], float) @ inc))) @ inc.T
    cq = np.asarray(p["Cq"], float)
    j = np.zeros((33, 33)); j[:10, 10:20] = np.eye(10)
    j[10:20, :10] = -(stiffness + cq @ ad) / mass[:, None]
    j[10:20, 10:20] = -np.diag(np.asarray(p["damping"], float) / mass)
    j[10:20, 20:30] = np.diag(1 / mass)
    j[10:20, 30:] = -cq / mass[:, None]
    j[20:30, 10:20] = -np.diag(np.asarray(p["gain"], float))
    j[20:30, 20:30] = -np.diag(np.asarray(p["rate"], float))
    j[30:, :10], j[30:, 10:20], j[30:, 20:30], j[30:, 30:] = bd, bw, bp, aq
    return j


def gauge_maps() -> tuple[np.ndarray, np.ndarray]:
    """Quotient only the physical common-angle gauge; retain all frequencies."""
    project, embed = np.zeros((32, 33)), np.zeros((33, 32))
    project[:9, :9] = np.eye(9); project[:9, 9] = -1
    project[9:19, 10:20] = np.eye(10); project[19:29, 20:30] = np.eye(10); project[29:, 30:] = np.eye(3)
    embed[:9, :9] = np.eye(9); embed[10:20, 9:19] = np.eye(10); embed[20:30, 19:29] = np.eye(10); embed[30:, 29:] = np.eye(3)
    return project, embed


def block_participation(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    """One four-block participation record per simple (including complex) mode."""
    values, right = np.linalg.eig(matrix); left = np.linalg.inv(right); normj = max(1., np.linalg.norm(matrix))
    groups = (slice(0, 9), slice(9, 19), slice(19, 29), slice(29, 32))
    records: list[dict[str, object]] = []
    for k, value in enumerate(values):
        sums = np.array([left[k, s] @ right[s, k] for s in groups])
        mags = np.abs(sums); normalized = mags / mags.sum()
        sep = min((abs(value-values[j]) for j in range(len(values)) if j != k), default=np.inf)
        residual = float(np.linalg.norm(matrix @ right[:, k] - value*right[:, k]) /
                         max(1., normj*np.linalg.norm(right[:, k])))
        records.append({"index": k, "eigenvalue": [float(value.real), float(value.imag)],
                        "block_sums": [[float(z.real), float(z.imag)] for z in sums],
                        "normalized_magnitudes": [float(z) for z in normalized], "sum_error": float(abs(sums.sum()-1)),
                        "separation": float(sep), "residual": residual,
                        "individually_unreliable": bool(sep <= 1e-7 or np.linalg.cond(right) > 1e10)})
    return values, right, left, records


def _field(model, states: np.ndarray) -> np.ndarray:
    import torch
    with torch.no_grad(): return model.field(torch.as_tensor(states, dtype=torch.float64)).detach().cpu().numpy()


def actual_jacobians(model) -> tuple[np.ndarray, dict[str, float], float]:
    """Autograd and two centered finite-difference checks of the live JointROM."""
    import torch
    zero = torch.zeros(33, dtype=torch.float64, requires_grad=True)
    auto = torch.autograd.functional.jacobian(lambda x: model.field(x[None])[0], zero).detach().cpu().numpy()
    f0 = _field(model, np.zeros((1, 33)))[0]; errors: dict[str, float] = {}
    for step in (1e-6, 5e-7):
        fd = np.empty((33, 33))
        for i in range(33):
            x = np.zeros((2, 33)); x[0, i], x[1, i] = step, -step
            y = _field(model, x); fd[:, i] = (y[0] - y[1]) / (2*step)
        errors[str(step)] = _relative(auto, fd)
    return auto, errors, float(np.abs(f0).max())


def storage_probe(model, base: MechanicalROM, p: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(65); states = rng.normal(size=(256, 33))
    states *= .1; states = np.vstack((np.zeros((1, 33)), states))
    field = _field(model, states)
    grad = base.energy_gradient(np.c_[states[:, :20], states[:, 20:30] + base.pm_star])
    nedge = p["angle_incidence"].shape[1]; scale = p["feature_scale"]
    phase = p["delta_star"] @ p["angle_incidence"]
    eta = states[:, :10] @ p["angle_incidence"] + phase
    angle = np.c_[np.sin(eta) - np.sin(phase), np.cos(eta) - np.cos(phase)]
    alpha = p["alpha"] * p["alpha_support"]; a = (angle / scale[20:20+2*nedge]) @ alpha
    force = (states[:, 30:] + a) @ p["Cq"].T
    damp = -np.sum(states[:, 10:20] * p["damping"] * states[:, 10:20], axis=1)
    governor = -np.sum(states[:, 20:30] * (p["rate"] / p["gain"]) * states[:, 20:30], axis=1)
    work = -np.sum(states[:, 10:20] * force, axis=1)
    lhs = np.sum(grad * field[:, :30], axis=1); rhs = damp + governor + work
    return {"states": states, "field": field, "gradient": grad, "dotH": lhs, "identity_rhs": rhs,
            "damping": damp, "governor": governor, "force_work": work, "algebraic_a": a}


def _m3_reference(base: MechanicalROM, values: np.ndarray) -> dict[str, object]:
    from audit_private_modes import full_jacobian, gauge_maps as m3_gauge
    project, embed = m3_gauge(); current = project @ full_jacobian(base) @ embed
    saved = np.load(VERIFIED / "M3_linearization.npz", allow_pickle=False)["matrix"]
    if current.shape != (29, 29): raise AssertionError("M3 quotient shape")
    return {"matrix": current, "saved": saved, "relative_max": _relative(saved, current), "positive_matches": []}


def run(output: Path) -> dict[str, object]:
    if output.exists(): raise FileExistsError(output)
    before = {"model": sha(MODEL/"model.pt"), "normalization": sha(MODEL/"normalization.npz"), "base": sha(BASE), "network": sha(NETWORK), "matrices": sha(MATRICES), "verified_evidence": sha(VERIFIED/"evidence.json")}
    if before["model"] != EXPECTED_MODEL or before["verified_evidence"] != EXPECTED_VERIFIED: raise AssertionError("frozen source hash")
    import torch
    torch.set_num_threads(1)
    matrices = dict(np.load(MATRICES, allow_pickle=False)); p = params(MODEL, matrices)
    if p["xi"].shape != (76, 3) or p["angle_incidence"].shape != (10, 26): raise AssertionError("actual feature contract")
    network = dict(np.load(NETWORK, allow_pickle=False)); network["observed_buses"] = np.unique(np.r_[network["load_buses"], network["generator_buses"]]); network["load_positions"] = np.searchsorted(network["observed_buses"], network["load_buses"])
    base = MechanicalROM.load(BASE); model = _construct(MODEL, base, network, "challenger")
    model.eval()
    analytic = analytic_jacobian(p); automatic, fd, equilibrium = actual_jacobians(model)
    project, embed = gauge_maps(); j32 = project @ analytic @ embed; gauge = np.r_[np.ones(10), np.zeros(23)]
    output.mkdir(parents=True)
    np.savez_compressed(output/"jacobian_precheck.npz", J33=analytic, J33_autograd=automatic, J32=j32, P=project, E=embed)
    values, right, left, blocks = block_participation(j32); residuals = np.array([np.linalg.norm(j32@right[:, i]-values[i]*right[:, i])/max(1., np.linalg.norm(j32)*np.linalg.norm(right[:, i])) for i in range(32)])
    m3 = _m3_reference(base, values)
    verified = json.loads((VERIFIED/"evidence.json").read_text()); full_modes = verified["fullorder_electromechanical_modes"]
    for i, v in enumerate(values):
        if v.imag > 1e-9:
            f = float(v.imag/(2*np.pi)); nearest = min(full_modes, key=lambda r: abs(float(r["frequency_hz"])-f))
            m3["positive_matches"].append({"index": i, "frequency_hz": f, "damping_ratio": float(-v.real/abs(v)), "nearest": nearest,
                                           "frequency_difference_hz": f-float(nearest["frequency_hz"]), "damping_difference": float(-v.real/abs(v))-float(nearest["damping_ratio"])})
    storage = storage_probe(model, base, p); storage_error = _relative(storage["dotH"], storage["identity_rhs"])
    max_real = float(values.real.max()); lyap: dict[str, object] = {"available": False}
    checks = {"equilibrium_max_abs": equilibrium, "analytic_autograd_relative_max": _relative(analytic, automatic), "fd_relative_max": fd,
              "gauge_relative_max": float(np.linalg.norm(analytic@gauge)/max(1., np.linalg.norm(analytic)*np.linalg.norm(gauge))),
              "PE_identity_max_abs": float(np.abs(project@embed-np.eye(32)).max()), "intertwining_max_abs": float(np.abs(project@analytic-j32@project).max()),
              "eigen_residual_max": float(residuals.max()), "block_sum_error_max": float(max(r["sum_error"] for r in blocks)), "M3_relative_max": m3["relative_max"], "storage_relative_max": storage_error}
    correct = bool(checks["equilibrium_max_abs"] <= 1e-10 and checks["analytic_autograd_relative_max"] <= 1e-9 and all(x <= 1e-6 for x in fd.values()) and checks["gauge_relative_max"] <= 1e-10 and checks["PE_identity_max_abs"] <= 1e-10 and checks["intertwining_max_abs"] <= 1e-10 and checks["eigen_residual_max"] <= 1e-9 and checks["block_sum_error_max"] <= 1e-9 and checks["M3_relative_max"] <= 1e-9 and checks["storage_relative_max"] <= 1e-9)
    if correct and max_real < -1e-6:
        lyap_p = solve_continuous_lyapunov(j32.T, -np.eye(32)); lyap_p = .5*(lyap_p+lyap_p.T)
        eigp = np.linalg.eigvalsh(lyap_p); resid = np.linalg.norm(j32.T@lyap_p+lyap_p@j32+np.eye(32))/max(1., np.linalg.norm(lyap_p)*np.linalg.norm(j32))
        lyap = {"available": True, "P": lyap_p, "eigenvalues": eigp, "condition": float(eigp.max()/eigp.min()), "relative_residual": float(resid), "positive_definite": bool(eigp.min() > 1e-10*max(1., eigp.max())), "residual_pass": bool(resid <= 1e-9)}
    screen = "verification_failure" if not correct else ("supported_condition" if max_real < -1e-6 else ("rejected_condition" if max_real > 1e-6 else "unresolved_marginal"))
    np.savez_compressed(output/"coupled_equilibrium.npz", J33=analytic, J33_autograd=automatic, J32=j32, P=project, E=embed, eigenvalues=values, right_eigenvectors=right, left_eigenvectors=left, eigen_residuals=residuals, M3_current=m3["matrix"], M3_verified=m3["saved"], storage_states=storage["states"], storage_field=storage["field"], storage_gradient=storage["gradient"], storage_dotH=storage["dotH"], storage_rhs=storage["identity_rhs"], storage_damping=storage["damping"], storage_governor=storage["governor"], storage_force_work=storage["force_work"], storage_a=storage["algebraic_a"], lyapunov_P=lyap.get("P", np.empty((0, 0)),), param_xi=p["xi"], param_support=p["support"], param_alpha=p["alpha"], param_alpha_support=p["alpha_support"], param_Cq=p["Cq"], param_mass=p["mass"], param_incidence=p["incidence"], param_delta_star=p["delta_star"], param_coupling=p["coupling"], param_damping=p["damping"], param_rate=p["rate"], param_gain=p["gain"], param_angle_incidence=p["angle_incidence"], param_feature_scale=p["feature_scale"], param_pm_star=base.pm_star)
    after = {k: sha(path) for k, path in {"model":MODEL/"model.pt", "normalization":MODEL/"normalization.npz", "base":BASE, "network":NETWORK, "matrices":MATRICES, "verified_evidence":VERIFIED/"evidence.json"}.items()}
    if after != before: raise AssertionError("frozen source changed during audit")
    spectrum = [{"index":i, "real":float(v.real), "imag":float(v.imag), "frequency_hz":float(abs(v.imag)/(2*np.pi)), "damping_ratio":None if abs(v)<1e-12 else float(-v.real/abs(v)), "residual":float(residuals[i])} for i,v in enumerate(values)]
    report = {"test_read":False, "source_sha256": {**before, "code":sha(Path(__file__)), "audit_long_latent_rollout":sha(ROOT/"experiments/audit_long_latent_rollout.py"), "evaluate_long_latent":sha(ROOT/"experiments/evaluate_long_latent.py"), "audit_private_modes":sha(ROOT/"experiments/audit_private_modes.py")}, "checks":checks, "correctness_pass":correct, "registered_raw_screen":screen, "spectrum":{"dominant_real":max_real,"eigenbasis_condition":float(np.linalg.cond(right)),"modes":spectrum,"complex_blocks":blocks}, "m3_reference":{"relative_max":m3["relative_max"],"positive_matches":m3["positive_matches"]}, "storage":{"n_states":int(len(storage["states"])),"positive_dotH_count":int(np.sum(storage["dotH"]>0)),"relative_identity_max":storage_error}, "conditional_lyapunov":{k:v for k,v in lyap.items() if k != "P"}}
    (output/"evidence.json").write_text(json.dumps(_jsonable(report), indent=2, allow_nan=False)); return report


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args(); run(args.output)


if __name__ == "__main__": main()
