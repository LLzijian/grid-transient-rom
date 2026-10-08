"""Loop 0076 prototype QA runner (synthetic and M3 assets only).

This runner deliberately performs no fitting and never opens event/H5 data.
It records numerical checks and trajectories; interpretation remains with the
root scientific agent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import subprocess
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

from structured_rom import (  # noqa: E402
    AnchoredGRUEncoder,
    DissipationSpec,
    ExchangeSpec,
    StructuredROM,
    numpy_field,
)
from structure_grammar import PrimitiveSpec  # noqa: E402
from kiki.models.private_pm import MechanicalROM  # noqa: E402


SEED = 76
DT_FINE = 0.005
DT_COARSE = 0.01
T_SYNTH = 10.0
T_HIDDEN = 5.0
TOL_IDENTITY = 1e-10
TOL_GRADIENT = 1e-10
TOL_FD = 1e-6


def _arr(x):
    return np.asarray(x, dtype=np.float64)


def _finite_max(x):
    x = _arr(x)
    return float(np.max(np.abs(x))) if x.size else 0.0


def _relerr(a, b, floor=1e-12):
    a, b = _arr(a), _arr(b)
    return float(np.max(np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), floor)))


def _normalized_max(residual, reference):
    residual, reference = _arr(residual), _arr(reference)
    return float(np.max(np.abs(residual) / np.maximum(1., np.abs(reference))))


def _structured_m3(base, n_q=3):
    return StructuredROM(
        mass=base.mass,
        D=base.incidence.T,
        coupling=base.coupling,
        delta_star=base.delta_star,
        damping=base.damping,
        rate=base.rate,
        gain=base.gain,
        n_q=n_q,
        scales=np.ones(10 + n_q),
        storage=(), exchange=(), dissipation=(), ports=(),
        n_p=10, jpi_p=np.zeros(10) + base.gain,
        rpp=base.rate * base.gain,
        trainable=False,
    ).double()


def _random_model(rng):
    n, np_, nq, ne = 3, 3, 3, 3
    D = rng.normal(size=(ne, n))
    D -= D.mean(axis=1, keepdims=True)
    delta_star = rng.normal(scale=.2, size=n)
    # Every state group is deliberately nonzero and mixed signed.
    storage = (
        PrimitiveSpec("sin", (0,), coefficient=.13, phi=.31),
        PrimitiveSpec("quadratic", (0,), coefficient=-.37, scale=(1.3,)),
        PrimitiveSpec("product", (1, 3), coefficient=.23, scale=(.9, 1.2)),
        PrimitiveSpec("state_sin", (2, 1), coefficient=-.19, scale=(1.1,), phi=.17),
        PrimitiveSpec("state_cos", (1, 2), coefficient=.07, scale=(.9,), phi=-.11),
        PrimitiveSpec("centered_cos", (0,), coefficient=.11, phi=-.23),
    )
    vdim = n + np_ + nq
    exchange = (
        ExchangeSpec(tuple(rng.normal(size=vdim)), tuple(rng.normal(size=vdim)), .31),
        ExchangeSpec(tuple(rng.normal(size=vdim)), tuple(rng.normal(size=vdim)), -.17),
    )
    dissipation = (
        DissipationSpec(tuple(rng.normal(size=vdim)), .27),
        DissipationSpec(tuple(rng.normal(size=vdim)), .13),
    )
    ports = (
        {"direction": tuple(rng.normal(size=vdim)), "terms": (
            {"kind": "linear", "index": 0, "coefficient": .21, "scale": .8},
            {"kind": "sin_difference", "index": 1, "coefficient": -.14},
        )},
        {"direction": tuple(rng.normal(size=vdim)), "terms": (
            {"kind": "linear", "index": 5, "coefficient": .18, "scale": 1.1},
            {"kind": "cos_difference", "index": 2, "coefficient": .09},
        )},
    )
    return StructuredROM(
        mass=np.array([.7, 1.4, 2.3]), D=D, coupling=np.array([.8, 1.2, .6]),
        delta_star=delta_star, damping=np.array([.12, .23, .31]),
        rate=np.array([.4, .7, .9]), gain=np.array([.8, 1.1, 1.5]),
        n_q=nq, scales=np.array([.8, 1.1, 1.4, .9, 1.2, .7]),
        storage=storage, exchange=exchange, dissipation=dissipation,
        ports=ports, n_p=np_, trainable=True,
    ).double()


def _rk4(model, z, dt):
    a = model.interface_field(z)
    b = model.interface_field(z + dt * a / 2)
    c = model.interface_field(z + dt * b / 2)
    d = model.interface_field(z + dt * c)
    return z + dt * (a + 2 * b + 2 * c + d) / 6


def _trajectory(model, initial, dt, horizon):
    out = [initial.copy()]
    z = torch.as_tensor(initial, dtype=torch.float64)
    for _ in range(round(horizon / dt)):
        z = _rk4(model, z, dt)
        out.append(z.detach().numpy().copy())
    return np.stack(out)


def _checks_random(model, rng):
    x = torch.as_tensor(rng.normal(size=(64, model.dim)), dtype=torch.float64)
    x.requires_grad_(True)
    with torch.no_grad():
        f0, e0, pv0, J0, R0 = model.field(torch.zeros_like(x), return_parts=True)
        f, e, pv, J, R = model.field(x, return_parts=True)
    H, eg = model.energy_and_gradient(x)
    ag = torch.autograd.grad(H.sum(), x, create_graph=False)[0]
    z = torch.zeros((1, model.dim), dtype=torch.float64)
    _, ez = model.energy_and_gradient(z)
    balance = (e * f).sum(-1) + (e @ R * e).sum(-1) - (pv * (e @ model.port_matrix())).sum(-1)
    diss_power = -(e.detach().numpy() @ R.detach().numpy() * e.detach().numpy()).sum(-1)
    checks = {
        "analytic_autograd_gradient_max_rel": _relerr(eg.detach().numpy(), ag.detach().numpy()),
        "f0_max_abs": _finite_max(f0),
        "energy_gradient_zero_max_abs": _finite_max(ez.detach().numpy()),
        "power_balance_max_abs": _finite_max(balance.detach().numpy()),
        "power_balance_normalized": _normalized_max(balance.detach().numpy(),
            torch.stack(((e*f).sum(-1).abs(), (e@R*e).sum(-1).abs(),
                         (pv*(e@model.port_matrix())).sum(-1).abs())).amax(0).detach().numpy()),
        "dissipation_max_positive_violation": float(max(0., float(np.max(diss_power)))),
    }
    # Gauge invariance and common-angle kinematics.
    shift = torch.ones_like(x[..., :model.n]) * .37
    xshift = x.detach().clone(); xshift[..., :model.n] += shift
    gauge_resid = model.field(xshift) - model.field(x.detach())
    kin_resid = f[..., :model.n] - x[..., model.n:2*model.n] / model.mass
    checks["angle_gauge_field_max_abs"] = _finite_max(gauge_resid.detach().numpy())
    checks["angle_gauge_field_normalized"] = _normalized_max(gauge_resid.detach().numpy(), model.field(x.detach()).detach().numpy())
    checks["kinematic_identity_max_abs"] = _finite_max(kin_resid.detach().numpy())
    checks["kinematic_identity_normalized"] = _normalized_max(kin_resid.detach().numpy(), f[..., :model.n].detach().numpy())
    # Central finite differences at both prescribed steps.
    fd = {}
    x0 = x.detach()[0].numpy()
    for eps in (1e-5, 1e-6):
        vals = []
        for row in x.detach().numpy():
            rowvals = []
            for j in range(model.dim):
                xp, xm = row.copy(), row.copy(); xp[j] += eps; xm[j] -= eps
                rowvals.append((float(model.energy(xp).detach()) - float(model.energy(xm).detach())) / (2 * eps))
            vals.append(rowvals)
        fd[str(eps)] = _relerr(np.asarray(vals), model.gradient(x.detach()).detach().numpy())
    checks["finite_difference_gradient_rel"] = fd
    # Verify each exchange group is zero-power and paired removal changes both terms.
    exchange_power = []
    Jbase, _ = model.matrices()
    for k, a, b in zip(model.exchange_kappa, model.exchange_a, model.exchange_b):
        aa = torch.zeros(model.dim, dtype=torch.float64); bb = aa.clone()
        aa[model.n:] = a / torch.linalg.vector_norm(a); bb[model.n:] = b / torch.linalg.vector_norm(b)
        Je = k * (torch.outer(aa, bb) - torch.outer(bb, aa))
        exchange_power.append(float(torch.max(torch.abs((e @ Je.T * e).sum(-1))).detach()))
    checks["exchange_power_max_abs"] = max(exchange_power, default=0.)
    # Removing one exchange group must remove both of its paired vector terms.
    with torch.no_grad():
        f_before = model.field(x.detach())
        raw_before = model.exchange_kappa[0].clone()
        aa = torch.zeros(model.dim, dtype=torch.float64); bb = aa.clone()
        aa[model.n:] = model.exchange_a[0] / torch.linalg.vector_norm(model.exchange_a[0])
        bb[model.n:] = model.exchange_b[0] / torch.linalg.vector_norm(model.exchange_b[0])
        je_before = raw_before * (torch.outer(aa, bb) - torch.outer(bb, aa))
        model.exchange_kappa[0].zero_()
        f_after = model.field(x.detach())
        model.exchange_kappa[0].copy_(raw_before)
    paired_resid = f_before - f_after - x.new_tensor(1.) * (e.detach() @ je_before.T)
    checks["exchange_pair_removal_max_abs"] = _finite_max(paired_resid.numpy())
    partner_rows = torch.any(torch.abs(je_before) > 1e-12, dim=1)
    checks["exchange_pair_both_nonzero"] = bool(torch.count_nonzero(partner_rows) >= 2)
    return checks


def _gradient_checks(model, rng):
    x = torch.as_tensor(rng.normal(size=(16, model.dim)), dtype=torch.float64)
    loss = model.field(x).square().mean() + model.energy(x).square().mean()
    loss.backward()
    families = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            g = p.grad
            families[name] = {"finite": bool(g is not None and torch.isfinite(g).all()), "norm": float(g.norm()) if g is not None else 0.0}
    enc = AnchoredGRUEncoder().double()
    hist = torch.as_tensor(rng.normal(size=(16, 11, 20)), dtype=torch.float64)
    eloss = enc(hist).square().mean(); eloss.backward()
    encoder = {name: {"finite": bool(p.grad is not None and torch.isfinite(p.grad).all()), "norm": float(p.grad.norm()) if p.grad is not None else 0.0} for name, p in enc.named_parameters() if p.requires_grad}
    return {"model": families, "encoder": encoder}


def _m3_check(base):
    model = _structured_m3(base, n_q=3)
    rng = np.random.default_rng(SEED + 1)
    z = rng.normal(size=(32, 33)); z[:, 30:] = 0.; z[:, :10] *= .1; z[:, 10:20] *= .2; z[:, 20:30] *= .2
    zi = torch.as_tensor(z, dtype=torch.float64)
    # MechanicalROM stores raw Pm; structured interface stores p=Pm-Pm_star.
    raw = z[:, :30].copy(); raw[:, 20:30] += base.pm_star
    source = base.field(raw, None); structured = model.interface_field(zi).detach().numpy()[:, :30]
    residual = structured - source
    return {"m3_physical_field_max_abs": _finite_max(residual), "m3_physical_field_normalized": _normalized_max(residual, source)}


def _gate_results(checks, metrics):
    grad = checks["gradients"]
    finite_grads = all(v["finite"] and v["norm"] > 0.0 for v in grad["model"].values()) and all(v["finite"] and v["norm"] > 0.0 for v in grad["encoder"].values())
    fd_ok = all(float(v) <= TOL_FD for v in checks["finite_difference_gradient_rel"].values())
    identities = {
        "kinematics": checks["kinematic_identity_normalized"] <= TOL_IDENTITY,
        "gauge": checks["angle_gauge_field_normalized"] <= TOL_IDENTITY,
        "origin": checks["f0_max_abs"] <= TOL_IDENTITY and checks["energy_gradient_zero_max_abs"] <= TOL_IDENTITY,
        "power_balance": checks["power_balance_normalized"] <= TOL_IDENTITY,
        "dissipation": checks["dissipation_max_positive_violation"] <= TOL_IDENTITY,
        "exchange": checks["exchange_power_max_abs"] <= TOL_IDENTITY and checks["exchange_pair_removal_max_abs"] <= TOL_IDENTITY and checks["exchange_pair_both_nonzero"],
        "gradient": checks["analytic_autograd_gradient_max_rel"] <= TOL_GRADIENT and fd_ok,
        "finite_active_gradients": finite_grads,
        "m3_field": checks["m3_physical_field_normalized"] <= TOL_IDENTITY,
        "export": checks["export_reload_field_max_abs"] <= TOL_IDENTITY and checks["torch_reload_field_max_abs"] <= TOL_IDENTITY,
    }
    syn = {
        "recovery_both_dt": metrics["recovery"]["finite"] and all(metrics["recovery"][f"{dt}_{k}_abs"] < .15 for dt in ("fine", "coarse") for k in ("angle", "speed")),
        "slip_both_dt": metrics["slip"]["finite"] and metrics["slip"]["fine_slip_crosses_pi"] and metrics["slip"]["coarse_slip_crosses_pi"],
        "hidden_finite": metrics["hidden"]["fine_finite"] and metrics["hidden"]["coarse_finite"],
        "hidden_shared_grid": metrics["hidden"]["fine_shared_grid_max_abs"] <= 1e-4,
        "hidden_work": all(metrics["hidden"]["work"][dt][k] > 1e-6 for dt in ("fine", "coarse") for k in ("exchange_transfer_integral_abs", "port_work_integral_abs")),
        "hidden_power_balance": all(metrics["hidden"]["work"][dt]["instantaneous_power_balance_max_abs"] <= TOL_IDENTITY for dt in ("fine", "coarse")),
    }
    return {"identity_gates": identities, "synthetic_gates": syn, "qa_passed": bool(all(identities.values()) and all(syn.values()))}


def _synthetic_cases():
    D = np.array([[1., -1.]])
    common = dict(mass=np.ones(2), D=D, coupling=np.ones(1), delta_star=np.zeros(2), damping=np.array([.2, .2]), rate=np.zeros(2), gain=np.ones(2), n_p=2, scales=np.ones(2), storage=(), exchange=(), dissipation=(), ports=(), trainable=False, jpi_p=np.zeros(2), rpp=np.zeros(2))
    recovery = StructuredROM(**common).double()
    slip = recovery
    hidden = StructuredROM(**{**common, "n_q": 1, "scales": np.ones(3), "storage": (), "exchange": (ExchangeSpec((1, 0, 0, 0, 0), (0, 0, 0, 0, 1), .3),), "dissipation": (DissipationSpec((0, 0, 0, 0, 1), .1),), "ports": ({"direction": (1, 0, 0, 0, 0), "terms": ({"kind": "linear", "index": 4, "coefficient": .05},)},)}).double()
    rec0 = np.array([.1, -.1, 0., 0., 0., 0.]); slip0 = rec0.copy(); slip0[2:4] = [3., -3.]
    hid0 = np.array([.1, -.1, 0., 0., 0., 0., .4])
    out = {}
    for name, m, initial, horizon in (("recovery", recovery, rec0, T_SYNTH), ("slip", slip, slip0, T_SYNTH), ("hidden", hidden, hid0, T_HIDDEN)):
        fine = _trajectory(m, initial, DT_FINE, horizon); coarse = _trajectory(m, initial, DT_COARSE, horizon)
        out[name] = {"fine": fine, "coarse": coarse}
    rel = out["recovery"]["fine"][:, 0] - out["recovery"]["fine"][:, 1]
    rel_slip = out["slip"]["fine"][:, 0] - out["slip"]["fine"][:, 1]
    def path_metrics(name, model, initial, horizon):
        fine, coarse = out[name]["fine"], out[name]["coarse"]
        rel_f = fine[:, 0] - fine[:, 1]; rel_c = coarse[:, 0] - coarse[:, 1]
        finite = bool(np.isfinite(fine).all() and np.isfinite(coarse).all())
        return {"finite": finite,
                "fine_final_angle": float(rel_f[-1]),
                "fine_final_speed": float(fine[-1, 2] - fine[-1, 3]),
                "coarse_final_angle": float(rel_c[-1]),
                "coarse_final_speed": float(coarse[-1, 2] - coarse[-1, 3]),
                "fine_speed_abs": float(abs(fine[-1, 2] - fine[-1, 3])),
                "fine_angle_abs": float(abs(rel_f[-1])),
                "coarse_speed_abs": float(abs(coarse[-1, 2] - coarse[-1, 3])),
                "coarse_angle_abs": float(abs(rel_c[-1])),
                "slip_crosses_pi": bool(max(np.max(np.abs(rel_f)), np.max(np.abs(rel_c))) > np.pi),
                "fine_slip_crosses_pi": bool(np.max(np.abs(rel_f)) > np.pi),
                "coarse_slip_crosses_pi": bool(np.max(np.abs(rel_c)) > np.pi)}

    hidden_work = {}
    for dt, key in ((DT_FINE, "fine"), (DT_COARSE, "coarse")):
        traj = out["hidden"][key]; exchange_transfer = []; port_work = []; balance = []
        for z in traj:
            zi = hidden.interface_to_state(torch.as_tensor(z, dtype=torch.float64))
            f, e, pv, J, R = hidden.field(zi, return_parts=True)
            aa = torch.zeros(hidden.dim, dtype=torch.float64); bb = aa.clone()
            aa[hidden.n:] = hidden.exchange_a[0] / torch.linalg.vector_norm(hidden.exchange_a[0]); bb[hidden.n:] = hidden.exchange_b[0] / torch.linalg.vector_norm(hidden.exchange_b[0])
            je = hidden.exchange_kappa[0] * (torch.outer(aa, bb) - torch.outer(bb, aa))
            fex = e @ je.T
            exchange_transfer.append(float((e[hidden.n] * fex[hidden.n]).detach()))
            port_work.append(float((pv * (e @ hidden.port_matrix())).sum().detach()))
            balance.append(float(((e * f).sum() + (e @ R * e).sum() - (pv * (e @ hidden.port_matrix())).sum()).detach()))
        exchange_integral = float(np.trapezoid(exchange_transfer, dx=dt))
        port_integral = float(np.trapezoid(port_work, dx=dt))
        hidden_work[key] = {"exchange_transfer_integral": exchange_integral,
                            "port_work_integral": port_integral,
                            "exchange_transfer_integral_abs": abs(exchange_integral),
                            "port_work_integral_abs": abs(port_integral),
                            "instantaneous_power_balance_max_abs": float(np.max(np.abs(balance)))}
    out["metrics"] = {"recovery": path_metrics("recovery", recovery, rec0, T_SYNTH), "slip": path_metrics("slip", slip, slip0, T_SYNTH), "hidden": {"fine_finite": bool(np.isfinite(out["hidden"]["fine"]).all()), "coarse_finite": bool(np.isfinite(out["hidden"]["coarse"]).all()), "fine_shared_grid_max_abs": _finite_max(out["hidden"]["fine"][::2] - out["hidden"]["coarse"]), "coarse_shared_grid_max_abs": _finite_max(out["hidden"]["fine"][::2] - out["hidden"]["coarse"]), "work": hidden_work}}
    return out


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--m3", type=Path, default=Path("private_data/task2_m3_0020/model.npz")); args = parser.parse_args()
    torch.set_num_threads(1); torch.set_default_dtype(torch.float64); torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    base = MechanicalROM.load(args.m3); model = _random_model(rng)
    checks = _checks_random(model, rng); gradients = _gradient_checks(model, rng); checks["gradients"] = gradients; checks.update(_m3_check(base))
    args.output.mkdir(parents=True, exist_ok=False); export_path = args.output / "model_numeric_export.json"; payload = model.export(export_path)
    x = rng.normal(size=(17, model.dim)); torch_field = model.field(torch.as_tensor(x, dtype=torch.float64)).detach().numpy(); np_field = numpy_field(payload, x); checks["export_reload_field_max_abs"] = _finite_max(torch_field - np_field); reload = StructuredROM.from_export(export_path).double(); checks["torch_reload_field_max_abs"] = _finite_max(torch_field - reload.field(torch.as_tensor(x, dtype=torch.float64)).detach().numpy())
    cases = _synthetic_cases(); np.savez_compressed(args.output / "synthetic_trajectories.npz", **{f"{k}_{r}": v[r] for k, v in cases.items() if isinstance(v, dict) and "fine" in v for r in ("fine", "coarse")})
    gates = _gate_results(checks, cases["metrics"])
    evidence = {"seed": SEED, "torch_seed": SEED, "test_read": False, "m3_path": str(args.m3), "checks": checks, "synthetic_metrics": cases["metrics"], "gates": gates, "thresholds": {"identity": TOL_IDENTITY, "gradient": TOL_GRADIENT, "fd": TOL_FD, "synthetic_shared_grid": 1e-4, "synthetic_work": 1e-6}, "export": payload.get("counts", {}), "provenance": {"command": " ".join(sys.argv), "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "grammar_sha256": hashlib.sha256((Path(__file__).resolve().parent / "structure_grammar.py").read_bytes()).hexdigest(), "structured_rom_sha256": hashlib.sha256((Path(__file__).resolve().parent / "structured_rom.py").read_bytes()).hexdigest(), "m3_sha256": hashlib.sha256(args.m3.read_bytes()).hexdigest()}}
    evidence["provenance"]["git_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    (args.output / "evidence.json").write_text(json.dumps(evidence, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(gates, indent=2), flush=True)


if __name__ == "__main__":
    main()
