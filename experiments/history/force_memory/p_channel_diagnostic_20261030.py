"""EXPLORATORY / POST-HOC: p-channel error decomposition (not pre-registered).

Read-only. Frozen A128 predictions + truth on train fit297 stable events; no training,
no model rollout. The frozen model's p dynamics is a fixed linear filter of omega:

    pdot_i = -jpi_p_i * omega_i - rpp_i * storage_base_i * p_i.

We split the p error into three parts at each 0.01 s output:
    p_pred - p_true = [p_pred - F(omega_pred)]     (integration/implementation)
                    + [F(omega_pred) - F(omega_true)]  (omega-induced)
                    + [F(omega_true) - p_true]         (p-model-form)
where F(omega) is the same RK4 filter integrated from the frozen p(0). All parts are
reported in the frozen normalized metric mean_i((.)/p_scale_i)^2. The parts are not
orthogonal; a sum-of-parts / total ratio is reported.

Scope: this is a diagnostic of the fixed p channel, not a mechanism proof, not a model
decision, and it was added after the pre-registered probe results were seen.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments"), str(ROOT / "src")]

MODEL = ROOT / "private_data/memory_transport_20261029/study01_prelaunch_retry1/study/A_30311/checkpoints/step000128/model.json"
PRED = ROOT / "private_data/memory_transport_20261029/study01_prelaunch_retry1/study/A_30311/fit000128"
SGM = ROOT / "private_data/state_gated_memory_20261028/study01"


def p_step(p, omega, dt, jpi, rpp_sb):
    def f(pp):
        return -jpi * omega - rpp_sb * pp
    k1 = f(p)
    k2 = f(p + dt * k1 / 2)
    k3 = f(p + dt * k2 / 2)
    k4 = f(p + dt * k3)
    return p + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def filter_p(omega, p0, jpi, rpp_sb, dt=0.01):
    p = np.array(p0, dtype=np.float64).copy()
    out = []
    for k in range(omega.shape[0]):
        p = p_step(p, omega[k], dt, jpi, rpp_sb)
        out.append(p.copy())
    return np.stack(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv)

    import state_gated_memory_study_20261028 as sgm
    import memory_transport_model_20261029 as mt

    _, events = sgm.checked_preparation(SGM, fit_only=True)
    by_id = {e["id"]: e for e in events}
    model = mt.MemoryTransportROM.from_export(MODEL).cpu().double().eval()
    jpi = model.jpi_p.detach().numpy()
    rpp_sb = (model.rpp * model.storage_base).detach().numpy()
    p_scale = model.p_scale.detach().numpy()

    terms = {"integration": [], "omega": [], "p_model": [], "total": []}
    recon_true = []
    for path in sorted(PRED.glob("predictions_*.npz")):
        z = np.load(path, allow_pickle=False)
        for eid, fine in zip(z["event_ids"], z["fine"]):
            e = by_id[str(eid)]
            if e["label"] != "stable" or len(e["x"]) - 11 < 800:
                continue
            n = 800
            omega_pred = fine[::2][1:n + 1, 10:20]
            p_pred = fine[::2][1:n + 1, 20:30]
            omega_true = e["x"][11:11 + n, 10:20]
            p_true = e["x"][11:11 + n, 20:30]
            p0 = fine[0, 20:30]
            f_pred = filter_p(omega_pred, p0, jpi, rpp_sb)
            f_true = filter_p(omega_true, p0, jpi, rpp_sb)
            scale = p_scale[None, :]

            def norm2(x):
                return float(np.mean((x / scale) ** 2, axis=0).mean())

            terms["integration"].append(norm2(p_pred - f_pred))
            terms["omega"].append(norm2(f_pred - f_true))
            terms["p_model"].append(norm2(f_true - p_true))
            terms["total"].append(norm2(p_pred - p_true))
            recon_true.append(float(np.linalg.norm(f_true - p_true) / max(np.linalg.norm(p_true), 1e-12)))

    summary = {"stable_events": len(terms["total"]), "terms": {}}
    for k in ("integration", "omega", "p_model", "total"):
        a = np.array(terms[k])
        summary["terms"][k] = {"mean": float(a.mean()), "median": float(np.median(a)), "p90": float(np.percentile(a, 90))}
    tot = np.array(terms["total"])
    summary["share_of_total_p_risk"] = {
        k: {"median_share": float(np.median(np.array(terms[k]) / tot)),
            "mean_share": float((np.array(terms[k]) / tot).mean())}
        for k in ("integration", "omega", "p_model")
    }
    summary["truth_p_reconstruction_rel"] = {"median": float(np.median(recon_true)), "max": float(max(recon_true))}
    s = np.array(terms["integration"]) + np.array(terms["omega"]) + np.array(terms["p_model"])
    summary["sum_of_parts_over_total"] = {"median": float(np.median(s / tot)), "max": float(np.max(s / tot))}
    summary["scope"] = ("EXPLORATORY post-hoc; frozen A128 + train fit297 stable; "
                        "frozen p coefficients; parts not orthogonal; not a mechanism claim")
    print(json.dumps(summary, indent=2, allow_nan=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
